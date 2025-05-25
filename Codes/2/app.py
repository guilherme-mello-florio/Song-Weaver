from flask import Flask, render_template, request, jsonify, url_for
import os
import tempfile
import uuid
import math
import logging
from collections import Counter
import tarfile # Pra extrair o modelo
import requests # Pra baixar o modelo

# music21 imports
from music21 import converter, tempo, pitch, key, environment, stream, note, chord, roman, common, meter, duration as m21duration

# Mido import
from mido import MidiFile as MidoMidiFile

# --- Tenta importar TensorFlow e Magenta ---
MAGENTA_AVAILABLE = False
tf = None
note_seq = None
configs = None
TrainedModel = None

try:
    import tensorflow as tf_actual
    import note_seq as ns_actual
    from magenta.models.music_vae import configs as cfg_actual
    from magenta.models.music_vae import TrainedModel as tm_actual

    tf = tf_actual
    note_seq = ns_actual
    configs = cfg_actual
    TrainedModel = tm_actual
    MAGENTA_AVAILABLE = True
    logging.info("TensorFlow e Magenta importados com sucesso.")
except ImportError as e:
    logging.error(f"Falha ao importar TensorFlow ou Magenta: {e}. Funcionalidade de geração estará desativada.")
    MAGENTA_AVAILABLE = False

app = Flask(__name__)

# --- Configurações ---
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
GENERATED_MIDI_DIR = os.path.join(app.static_folder, 'generated_midi')
os.makedirs(GENERATED_MIDI_DIR, exist_ok=True)

MODEL_CONFIG_NAME = 'cat-mel_2bar_big'
music_vae_model = None

# Caminho pro cache de modelos do Magenta e URL de download
MAGENTA_CACHE_DIR = os.path.expanduser(os.path.join("~", ".magenta", "models"))
MODEL_DOWNLOAD_URL = f"http://download.magenta.tensorflow.org/models/music_vae/checkpoints/{MODEL_CONFIG_NAME}.tar.gz"
MODEL_LOCAL_DIR = os.path.join(MAGENTA_CACHE_DIR, MODEL_CONFIG_NAME)

EXPECTED_CHECKPOINT_FILE_IN_MODEL_DIR = os.path.join(MODEL_LOCAL_DIR, "checkpoint")


logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# --- Funções Auxiliares Magenta ---

def _download_and_extract_model():
    """Baixa e extrai o modelo MusicVAE se não existir localmente."""
    if os.path.exists(EXPECTED_CHECKPOINT_FILE_IN_MODEL_DIR):
        app.logger.info(f"Checkpoint do modelo já encontrado em: {MODEL_LOCAL_DIR}")
        return True

    app.logger.info(f"Checkpoint não encontrado. Tentando baixar de {MODEL_DOWNLOAD_URL} para {MODEL_LOCAL_DIR}...")
    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
    
    temp_tar_path = os.path.join(MODEL_LOCAL_DIR, f"{MODEL_CONFIG_NAME}.tar.gz")

    try:
        response = requests.get(MODEL_DOWNLOAD_URL, stream=True)
        response.raise_for_status() # Levanta erro pra códigos HTTP ruins (4xx ou 5xx)
        with open(temp_tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        app.logger.info(f"Modelo baixado: {temp_tar_path}")

        app.logger.info(f"Extraindo modelo para {MODEL_LOCAL_DIR}...")
        with tarfile.open(temp_tar_path, "r:gz") as tar:
            tar.extractall(path=MODEL_LOCAL_DIR)
        app.logger.info("Modelo extraído com sucesso.")
        os.remove(temp_tar_path) # Remove arquivo .tar.gz após extração
        
        if not os.path.exists(EXPECTED_CHECKPOINT_FILE_IN_MODEL_DIR):
            app.logger.error(f"Arquivo 'checkpoint' não encontrado em {MODEL_LOCAL_DIR} após extração.")
            return False
        return True
        
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Falha no download do modelo: {e}")
        return False
    except tarfile.TarError as e:
        app.logger.error(f"Falha ao extrair o modelo: {e}")
        return False
    except Exception as e:
        app.logger.error(f"Erro inesperado durante download/extração: {e}", exc_info=True)
        return False


def load_music_vae_model():
    global music_vae_model
    if not MAGENTA_AVAILABLE:
        app.logger.error("Magenta/TensorFlow não disponíveis, não é possível carregar o modelo.")
        music_vae_model = "ERROR"
        return

    if music_vae_model is None:
        app.logger.info(f"Carregando modelo MusicVAE: {MODEL_CONFIG_NAME}...")
        try:
            if not _download_and_extract_model():
                raise FileNotFoundError(f"Falha ao garantir que o modelo {MODEL_CONFIG_NAME} esteja disponível localmente.")

            if MODEL_CONFIG_NAME not in configs.CONFIG_MAP:
                raise ValueError(f"Configuração MusicVAE desconhecida: {MODEL_CONFIG_NAME}")

            config = configs.CONFIG_MAP[MODEL_CONFIG_NAME]
            config.data_converter.max_input_length = 32
            
            app.logger.info(f"Instanciando TrainedModel com checkpoint_dir_or_path='{MODEL_LOCAL_DIR}'")
            music_vae_model = TrainedModel(
                config,
                batch_size=1,
                checkpoint_dir_or_path=MODEL_LOCAL_DIR # Aponta pro DIRETÓRIO do checkpoint
            )
            app.logger.info("Modelo MusicVAE carregado com sucesso.")

        except Exception as e:
            app.logger.error(f"Erro geral ao carregar modelo MusicVAE: {e}", exc_info=True)
            music_vae_model = "ERROR"
    elif music_vae_model == "ERROR":
        app.logger.warning("Modelo MusicVAE falhou ao carregar anteriormente.")


def convert_midi_to_notesequence(midi_file_path):
    if not MAGENTA_AVAILABLE:
        app.logger.error("Não é possível converter MIDI para NoteSequence: Magenta/note-seq não disponível.")
        return None
    try:
        # Assegurar que note_seq esteja disponível se MAGENTA_AVAILABLE for True
        ns = note_seq.midi_file_to_note_sequence(midi_file_path)
        if not ns.notes:
            app.logger.warning(f"Nenhuma nota encontrada no arquivo MIDI: {midi_file_path}")
            # Retorna NoteSequence vazio válido em vez de None, pode simplificar o fluxo
            return note_seq.protobuf.music_pb2.NoteSequence()
        if ns.total_time == 0 and ns.notes:
            ns.total_time = max(note.end_time for note in ns.notes)
        app.logger.info(f"Convertido com sucesso {midi_file_path} para NoteSequence.")
        return ns
    except Exception as e: 
        app.logger.error(f"Erro ao converter MIDI {midi_file_path} para NoteSequence: {e}", exc_info=True)
        return None

def prepare_primer_sequence(full_notesequence, num_bars_primer=2):
    if not MAGENTA_AVAILABLE:
        app.logger.error("Não é possível preparar primer: Magenta/note-seq não disponível.")
        return None
    if not full_notesequence or not full_notesequence.notes:
        app.logger.warning("Não é possível preparar primer de NoteSequence vazio ou sem notas.")
        return None
    try:
        qpm = note_seq.constants.DEFAULT_QUARTERS_PER_MINUTE
        if full_notesequence.tempos:
            relevant_tempos = [t for t in full_notesequence.tempos if t.time <= full_notesequence.total_time and t.qpm > 0]
            if relevant_tempos: qpm = relevant_tempos[-1].qpm
            elif full_notesequence.tempos and full_notesequence.tempos[0].qpm > 0: qpm = full_notesequence.tempos[0].qpm
        
        numerator, denominator = 4, 4
        if full_notesequence.time_signatures:
            relevant_ts = [ts for ts in full_notesequence.time_signatures if ts.time <= full_notesequence.total_time]
            ts_to_use = relevant_ts[-1] if relevant_ts else (full_notesequence.time_signatures[0] if full_notesequence.time_signatures else None)
            if ts_to_use: numerator, denominator = ts_to_use.numerator, ts_to_use.denominator

        seconds_per_bar = (60.0 / qpm) * numerator * (4.0 / denominator) if qpm > 0 and denominator > 0 else 0
        primer_duration_seconds = num_bars_primer * seconds_per_bar if seconds_per_bar > 0 else 5.0
        start_time = max(0.0, full_notesequence.total_time - primer_duration_seconds)
        end_time = full_notesequence.total_time
        if start_time >= end_time and full_notesequence.total_time > 0: start_time = 0.0
        
        primer_ns = note_seq.extract_subsequence(full_notesequence, start_time, end_time)
        if not primer_ns.notes: app.logger.warning("Primer extraído não contém notas."); return None
        
        steps_per_second = note_seq.steps_per_second_for_qpm(qpm) if qpm > 0 else 60
        quantized_primer_ns = note_seq.quantize_note_sequence(primer_ns, steps_per_second)
        if not quantized_primer_ns.notes: app.logger.warning("Primer quantizado não contém notas."); return None
        
        if quantized_primer_ns.total_time == 0 and quantized_primer_ns.notes:
            quantized_primer_ns.total_time = max(n.quantized_end_step for n in quantized_primer_ns.notes) / (steps_per_second if steps_per_second > 0 else 1.0) # Evitar divisão por zero
        
        quantized_primer_ns.ticks_per_quarter = full_notesequence.ticks_per_quarter or note_seq.constants.STANDARD_PPQ
        del quantized_primer_ns.tempos[:]; del quantized_primer_ns.time_signatures[:]
        
        primer_tempo_obj = quantized_primer_ns.tempos.add()
        primer_tempo_obj.qpm = qpm
        primer_tempo_obj.time = 0
        
        primer_ts_obj = quantized_primer_ns.time_signatures.add()
        primer_ts_obj.numerator = numerator
        primer_ts_obj.denominator = denominator
        primer_ts_obj.time = 0
        
        app.logger.info(f"Sequência primer preparada (quantizada) de {start_time:.2f}s a {end_time:.2f}s.")
        return quantized_primer_ns
    except Exception as e:
        app.logger.error(f"Erro ao preparar sequência primer: {e}", exc_info=True)
        return None

def generate_continuation_sequence(model, primer_sequence, full_notesequence_ref, length_seconds=8, temperature=0.6):
    if not MAGENTA_AVAILABLE: return None
    if not primer_sequence or not primer_sequence.notes or model is None or model == "ERROR": return None
    try:
        qpm = primer_sequence.tempos[0].qpm if primer_sequence.tempos and primer_sequence.tempos[0].qpm > 0 else note_seq.constants.DEFAULT_QUARTERS_PER_MINUTE
        steps_per_second = note_seq.steps_per_second_for_qpm(qpm) if qpm > 0 else 60.0 # Evitar divisão por zero
        model_output_steps = model.config.data_converter.max_input_length
        if not model_output_steps or model_output_steps <=0: model_output_steps = 32 # fallback seguro
        
        num_requested_continuation_steps = int(length_seconds * steps_per_second)
        num_chunks_needed = math.ceil(num_requested_continuation_steps / model_output_steps)
        
        app.logger.info(f"Gerando {num_chunks_needed} chunks para approx {length_seconds}s de continuação.")

        generated_sequences = model.sample(
            n=num_chunks_needed,
            length=model_output_steps,
            temperature=temperature,
            primer_sequence=primer_ns # Passar primer_ns aqui
        )
        
        if not generated_sequences: raise ValueError("Sample falhou em retornar sequências.")
        
        # Concatena chunks gerados, ajustando os tempos
        all_notes = []
        current_time_offset = primer_sequence.total_time

        for i, chunk_ns in enumerate(generated_sequences):
            if not chunk_ns.notes: continue

        # ---- VOLTANDO PARA INTERPOLATE ----
        num_interpolate_steps = num_chunks_needed + 1 # Um a mais para o primer
        app.logger.info(f"Usando INTERPOLATE: {num_interpolate_steps} passos de interpolação.")

        z_list, _, _ = model.encode([primer_sequence])
        if not z_list.size > 0: raise ValueError("Encode falhou.")

        interpolated_sequences = model.interpolate(
            z_start=z_list, 
            z_end=z_list, # Interpola primer para ele mesmo, pegando passos intermed
            num_steps=num_interpolate_steps,
            length=model_output_steps, 
            temperature=temperature
        )

        if not interpolated_sequences or len(interpolated_sequences) < 2:
             raise ValueError("Interpolação não retornou sequências suficientes.")

        # O primeiro é o primer reconstruído, pegamos os
        continuation_chunks = interpolated_sequences[1:] 
        # ---- FIM DA LÓGICA DE INTERPOLATE ----

        if not any(chunk.notes for chunk in continuation_chunks):
            app.logger.warning("Nenhum dos chunks de continuação gerados continha notas.")
            return None

        final_continuation_ns = note_seq.concatenate_sequences(continuation_chunks)
        if not final_continuation_ns.notes:
            app.logger.warning("Concatenação dos chunks de continuação resultou em sequência sem notas.")
            return None

        # Desloca continuação pra começar pos primer
        shifted_continuation_ns = note_seq.sequences_lib.shift_sequence_times(
            final_continuation_ns, primer_sequence.total_time
        )
        
        # Extrai duração desejada
        target_end_time = primer_sequence.total_time + length_seconds
        final_ns_trimmed = note_seq.extract_subsequence(
            shifted_continuation_ns, 
            start_time=primer_sequence.total_time, 
            end_time=target_end_time
        )

        if not final_ns_trimmed.notes:
            app.logger.warning("Continuação após trim não contém notas.")
            return None

        # Preserva QPM e TS do primer ou original
        del final_ns_trimmed.tempos[:]; del final_ns_trimmed.time_signatures[:]
        tempo_obj = final_ns_trimmed.tempos.add()
        tempo_obj.qpm = qpm
        tempo_obj.time = 0 # Relativo ao início nova sequência ja desloc

        ts_obj = final_ns_trimmed.time_signatures.add()
        if primer_sequence.time_signatures:
            ts_obj.numerator = primer_sequence.time_signatures[0].numerator
            ts_obj.denominator = primer_sequence.time_signatures[0].denominator
        elif full_notesequence_ref and full_notesequence_ref.time_signatures:
            ts_obj.numerator = full_notesequence_ref.time_signatures[0].numerator
            ts_obj.denominator = full_notesequence_ref.time_signatures[0].denominator
        else:
            ts_obj.numerator = 4
            ts_obj.denominator = 4
        ts_obj.time = 0
        final_ns_trimmed.ticks_per_quarter = primer_sequence.ticks_per_quarter or note_seq.constants.STANDARD_PPQ
        
        app.logger.info(f"Sequência de continuação gerada com {len(final_ns_trimmed.notes)} notas.")
        return final_ns_trimmed
    except Exception as e:
        app.logger.error(f"Erro durante a geração MusicVAE: {e}", exc_info=True)
        return None

def save_notesequence_to_midi(notesequence, output_dir):
    if not MAGENTA_AVAILABLE: return None
    if not notesequence or not notesequence.notes: return None
    try:
        output_filename = f"continuation_{uuid.uuid4().hex[:10]}.mid"
        output_path = os.path.join(output_dir, output_filename)
        note_seq.note_sequence_to_midi_file(notesequence, output_path)
        app.logger.info(f"MIDI gerado salvo em: {output_path}")
        return os.path.join(os.path.basename(output_dir), output_filename)
    except Exception as e:
        app.logger.error(f"Erro ao salvar NoteSequence para MIDI: {e}", exc_info=True)
        return None

def is_initial_midi_valid(file_stream):
    try:
        file_stream.seek(0); MidoMidiFile(file=file_stream); file_stream.seek(0)
        return True
    except Exception: return False

def analyze_midi_with_music21(file_path):
    results = { "bpm": "N/A", "key": "N/A", "time_signature": "N/A", "num_bars": "N/A", "melodic_range": "N/A", "chord_complexity": "N/A", "rhythmic_density": "N/A", "form_structure": "Ainda não implementado", "harmonic_progression_preview": "N/A", "rhythmic_pattern_summary": "N/A", "ai_analysis_text": "Aguardando dados da análise..." }
    try:
        us = environment.UserSettings(); us['autoDownload'] = 'allow'; us['showFormat'] = 'text'; us['graphicsPath'] = None
        import warnings; warnings.filterwarnings('ignore', message=".*"); s = converter.parse(file_path)
        if not s: results["ai_analysis_text"] = "Não foi possível carregar o arquivo com music21."; return results
        
        bpm_values = [el.number for el in s.flat.getElementsByClass(tempo.MetronomeMark) if el.number is not None]
        if bpm_values: results["bpm"] = round(sum(bpm_values) / len(bpm_values)) if len(bpm_values) > 1 else round(bpm_values[0])
        
        key_obj = None
        try:
            key_obj = s.analyze('key')
            if key_obj and key_obj.tonic: results["key"] = f"{key_obj.tonic.name.replace('-', '♭').replace('#', '♯')} {('Maior' if key_obj.mode == 'major' else 'Menor' if key_obj.mode == 'minor' else key_obj.mode if key_obj.mode else '')}".strip()
        except Exception: results["key"] = "N/A"
        
        ts_search = s.flat.getElementsByClass(meter.TimeSignature); results["time_signature"] = ts_search[0].ratioString if ts_search and ts_search[0].ratioString else "4/4"
        
        measures = list(s.recurse().getElementsByClass(stream.Measure))
        if measures: results["num_bars"] = max(m.number for m in measures if m.number is not None) if any(m.number for m in measures) else len(measures)
        elif s.highestTime > 0 and results["time_signature"] != "N/A":
            try: 
                ts_obj = meter.TimeSignature(results["time_signature"])
                if ts_obj.barDuration.quarterLength > 0:
                    results["num_bars"] = math.ceil(s.highestTime / ts_obj.barDuration.quarterLength)
            except Exception: pass # Deixa N/A se falhar
        
        all_pitches = list(s.flat.pitches)
        if all_pitches:
            min_ps, max_ps = min(p.ps for p in all_pitches), max(p.ps for p in all_pitches); oct_span = (max_ps - min_ps) / 12.0
            if min_ps == max_ps: results["melodic_range"] = "Tom único"
            else: results["melodic_range"] = f"~{round(oct_span) if oct_span >=1 else '< 1'} Oitava{'s' if oct_span >=1.5 else ''}"
        else: results["melodic_range"] = "Nenhuma nota"

        key_for_roman_analysis = key_obj
        try:
            chordified_stream = s.chordify(); chords_list = list(chordified_stream.flat.getElementsByClass(chord.Chord))
            if chords_list:
                qualities = set(c.quality for c in chords_list if hasattr(c, 'quality')); num_q = len(qualities)
                results["chord_complexity"] = "Mínima" if num_q == 0 and chords_list else "Simples" if num_q <= 2 else "Moderada" if num_q <= 4 else "Complexa"
                
                distinct_chords_preview, temp_seen_reprs, prog_common, prog_roman = [], set(), [], []
                for ch_obj in chords_list:
                    repr_to_check = ch_obj.pitchedCommonName
                    if key_for_roman_analysis and key_for_roman_analysis.tonic:
                        try: repr_to_check = roman.romanNumeralFromChord(ch_obj, key_for_roman_analysis).figure
                        except Exception: pass
                    if repr_to_check not in temp_seen_reprs and len(distinct_chords_preview) < 4:
                        distinct_chords_preview.append(ch_obj); temp_seen_reprs.add(repr_to_check)
                
                for ch_p in distinct_chords_preview:
                    cn = ch_p.pitchedCommonName.replace('-', '♭').replace('#', '♯'); prog_common.append(cn)
                    try:
                        if key_for_roman_analysis and key_for_roman_analysis.tonic: prog_roman.append(roman.romanNumeralFromChord(ch_p, key_for_roman_analysis).figure)
                        else: prog_roman.append(cn)
                    except Exception: prog_roman.append(cn)
                results["harmonic_progression_preview"] = " → ".join(prog_roman if prog_roman and prog_roman != prog_common else prog_common) if prog_common else "N/A"
            else: results["chord_complexity"] = "Nenhuma"
        except Exception as e: app.logger.error(f"Erro análise harmônica: {e}", exc_info=True); results["chord_complexity"]="Erro na Análise"; results["harmonic_progression_preview"]="N/A"

        notes_rests_count = len(list(s.flat.notesAndRests))
        num_bars_val_for_density = results.get("num_bars")
        if isinstance(num_bars_val_for_density, int) and num_bars_val_for_density > 0 and notes_rests_count > 0:
            density = notes_rests_count / num_bars_val_for_density
            results["rhythmic_density"] = "Baixa" if density < 8 else "Média" if density < 20 else "Alta"
        elif notes_rests_count == 0: results["rhythmic_density"] = "Nenhuma nota"
        else: results["rhythmic_density"] = "N/A" # Se num_bars não for int ou for 0
        
        note_durations_ql = [n.duration.quarterLength for n in s.flat.notes if hasattr(n, 'duration') and n.duration is not None and n.duration.quarterLength is not None]
        if note_durations_ql:
            try:
                common_durations_counter = Counter(note_durations_ql).most_common(1)
                if common_durations_counter:
                    most_common_ql_value = common_durations_counter[0][0]
                    d_obj = m21duration.Duration(quarterLength=most_common_ql_value)
                    type_map = {'whole': 'Semibreve', 'half': 'Mínima', 'quarter': 'Semínima', 'eighth': 'Colcheia', '16th': 'Semicolcheia', '32nd': 'Fusa', '64th': 'Semifusa', 'longa': 'Longa', 'breve': 'Breve', 'duplex-maxima': 'Máxima Dupla', 'maxima': 'Máxima', 'zero': 'Duração Zero'}
                    duration_name = type_map.get(d_obj.type, f"Tipo '{d_obj.type}'")
                    dots_str = "." * d_obj.dots if d_obj.dots > 0 else ""
                    results["rhythmic_pattern_summary"] = f"Predominância de {duration_name}{dots_str}s"
            except Exception as e_rhythm: app.logger.warning(f"Falha ao analisar padrão rítmico: {e_rhythm}")
        elif len(list(s.flat.notes)) > 0: results["rhythmic_pattern_summary"] = "Durações não encontradas"
        else: results["rhythmic_pattern_summary"] = "Nenhuma nota"
        
        analysis_parts = []
        if results.get("key") not in ["N/A", "Erro na análise"]: analysis_parts.append(f"A tonalidade principal parece ser {results['key']}.")
        if results.get("bpm") not in ["N/A", "Erro na análise"]: analysis_parts.append(f"O andamento médio é de aproximadamente {results['bpm']} BPM.")
        if results.get("time_signature") not in ["N/A", "Erro na análise"]: analysis_parts.append(f"Utiliza um compasso de {results['time_signature']}.")
        if results.get("harmonic_progression_preview") not in ["N/A", "Erro na análise"] and results["harmonic_progression_preview"]: analysis_parts.append(f"A progressão harmônica inicial observada é: {results['harmonic_progression_preview']}.")
        elif results.get("chord_complexity") not in ["Nenhuma", "Erro na Análise", "N/A", "Mínima", "Erro na análise"]: analysis_parts.append(f"A complexidade harmônica é classificada como {results['chord_complexity'].lower()}.")
        if results.get("rhythmic_pattern_summary") not in ["N/A", "Nenhuma nota", "Durações não encontradas", "Erro na análise"]: analysis_parts.append(f"{results['rhythmic_pattern_summary']}.")
        elif results.get("rhythmic_density") not in ["N/A", "Nenhuma nota", "Erro na análise"]: analysis_parts.append(f"A densidade rítmica é {results['rhythmic_density'].lower()}.")
        if results.get("melodic_range") not in ["N/A", "Nenhuma nota", "Tom único", "< 1 Oitava", "Erro na análise"]: analysis_parts.append(f"A melodia se estende por {results['melodic_range'].lower()}.")
        
        if analysis_parts: results["ai_analysis_text"] = " ".join(analysis_parts)
        elif not list(s.flat.notes): results["ai_analysis_text"] = "O arquivo MIDI não contém notas."
        else: results["ai_analysis_text"] = "Análise textual não disponível."

    except Exception as e:
        app.logger.error(f"Erro geral em analyze_midi_with_music21: {e}", exc_info=True)
        for k in results: results[k] = "Erro na análise" # Garante que todos campos mostrem erro
        results["ai_analysis_text"] = f"Erro crítico music21: {str(e)[:100]}"
    return results

# --- Rotas Flask --- 
@app.route('/')
def home():
    page_data = {"logo_title": "Music VAE Assistant"} # Simplificado para exemplo
    analysis_results_data = {"bpm": "...", "key": "...", "bars": "..."}
    ai_analysis_text = "Importe um arquivo MIDI para ver a análise."
    composition_stats_data = {
        "melodic_range": "...", "chord_complexity": "...",
        "rhythmic_density": "...", "form_structure": "..."
    }
    return render_template('index.html', page_data=page_data, analysis_results=analysis_results_data,
                           ai_analysis=ai_analysis_text, composition_stats=composition_stats_data)

@app.route('/upload_midi', methods=['POST'])
def upload_midi_file():
    if 'midi_file' not in request.files:
        return jsonify({"status": "error", "message": "Nenhum arquivo enviado."}), 400
    file = request.files['midi_file']
    if not file.filename:
        return jsonify({"status": "error", "message": "Nenhum arquivo selecionado."}), 400
    
    _, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() not in ['.mid', '.midi']:
        return jsonify({"status": "error", "message": "Extensão de arquivo inválida. Use .mid ou .midi."}), 400
    
    safe_filename_base = "".join(c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in ('-', '_')).rstrip()
    if not safe_filename_base: safe_filename_base = "midi_upload"
    unique_filename = f"{safe_filename_base}_{uuid.uuid4().hex[:8]}{file_extension.lower()}"
    saved_midi_path = os.path.join(UPLOADS_DIR, unique_filename)
    
    try:
        file.save(saved_midi_path) # Salva arquivo primeiro
        app.logger.info(f"Arquivo salvo em: {saved_midi_path}")

        # Valida arquivo salvo
        with open(saved_midi_path, 'rb') as f_validate:
            if not is_initial_midi_valid(f_validate):
                os.unlink(saved_midi_path) # Remove arquivo inválido
                return jsonify({"status": "error", "original_filename": file.filename, "message": "Arquivo MIDI não parece ser válido (validação inicial)."}), 400
        
        analysis_data = analyze_midi_with_music21(saved_midi_path)
        if "Erro crítico" in analysis_data.get("ai_analysis_text", ""): # Checa erro crítico analise
             return jsonify({"status": "error", "original_filename": file.filename, "message": analysis_data["ai_analysis_text"]}), 500

        return jsonify({
            "status": "success", "filename": unique_filename,
            "original_filename": file.filename, "message": "Análise MIDI concluída.",
            "analysis": analysis_data
        }), 200
    except Exception as e:
        app.logger.error(f"Erro no processo de upload ou análise para {file.filename}: {e}", exc_info=True)
        if os.path.exists(saved_midi_path): # Tenta remover se algo deu ruim apos salvar
            try: os.unlink(saved_midi_path)
            except OSError: pass
        return jsonify({"status": "error", "original_filename": file.filename, "message": f"Erro interno no processamento do arquivo: {str(e)}"}), 500

@app.route('/generate_continuation', methods=['POST'])
def generate_continuation_route():
    if not MAGENTA_AVAILABLE:
        return jsonify({"status": "error", "message": "Funcionalidade de geração indisponível (Magenta/TF não carregado)."}), 501
    
    load_music_vae_model() # Tenta (re)carregar modelo se precsar
    if music_vae_model is None or music_vae_model == "ERROR":
        return jsonify({"status": "error", "message": "Modelo MusicVAE não está disponível. Não é possível gerar."}), 503
    
    data = request.get_json()
    filename = data.get('filename') if data else None
    if not filename:
        return jsonify({"status": "error", "message": "Nome do arquivo não fornecido no corpo da requisição."}), 400

    # Segurança: usar os.path.basename para evitar path traversal
    original_midi_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(original_midi_path):
        app.logger.error(f"Arquivo MIDI original não encontrado: {original_midi_path}")
        return jsonify({"status": "error", "message": "Arquivo MIDI original para geração não encontrado no servidor."}), 404

    full_ns = convert_midi_to_notesequence(original_midi_path)
    if not full_ns or not full_ns.notes:
        return jsonify({"status": "error", "message": "Falha ao converter MIDI original para NoteSequence ou MIDI sem notas."}), 500
    
    primer_ns = prepare_primer_sequence(full_ns)
    if not primer_ns or not primer_ns.notes:
        return jsonify({"status": "error", "message": "Falha ao preparar sequência primer (pode ser muito curta ou inválida)."}), 500

    try:
        length_seconds = float(data.get('length_seconds', 8.0))
        temperature = float(data.get('temperature', 0.6))
        if not (0.1 <= temperature <= 1.5): temperature = 0.6
        if not (1.0 <= length_seconds <= 30.0): length_seconds = 8.0
    except ValueError:
        return jsonify({"status": "error", "message": "Parâmetros de geração inválidos (length/temperature devem ser números)."}), 400

    continuation_ns = generate_continuation_sequence(music_vae_model, primer_ns, full_ns, length_seconds, temperature)
    if not continuation_ns or not continuation_ns.notes:
        return jsonify({"status": "error", "message": "Geração MusicVAE falhou ou não produziu notas."}), 500
    
    relative_midi_path = save_notesequence_to_midi(continuation_ns, GENERATED_MIDI_DIR)
    if not relative_midi_path:
        return jsonify({"status": "error", "message": "Falha ao salvar arquivo MIDI gerado."}), 500
    
    try:
        midi_url = url_for('static', filename=relative_midi_path, _external=False)
    except Exception as e_url:
        app.logger.error(f"Erro ao gerar URL para {relative_midi_path}: {e_url}")
        return jsonify({"status": "error", "message": "Não foi possível gerar URL do arquivo MIDI."}), 500
        
    return jsonify({
        "status": "success", "message": "Continuação gerada com sucesso.",
        "midi_filename": os.path.basename(relative_midi_path), "midi_url": midi_url
    }), 200

# --- Execução Principal ---
if __name__ == '__main__':
    try:
        us = environment.UserSettings()
        us['autoDownload'] = 'allow'
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='music21') # Pra warnings do music21
        warnings.filterwarnings('ignore', message="midi.*") # Warnings gerais de MIDI music21
        app.logger.info("Configurações do music21 aplicadas.")
    except Exception as e_env:
        app.logger.warning(f"Não foi possível definir configurações globais do music21: {e_env}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) # Porta 5000 explícita