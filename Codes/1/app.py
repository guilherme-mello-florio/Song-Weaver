from flask import Flask, render_template, request, jsonify
import os
import tempfile # Para salvar o arquivo temporariamente para music21

# music21 imports
from music21 import converter, tempo, pitch, key, environment, stream, note, chord, roman, common, meter, duration as m21duration
from collections import Counter

# Mido para uma validação inicial rápida (opcional, music21 também valida ao parsear)
from mido import MidiFile as MidoMidiFile

app = Flask(__name__)

# --- Validação MIDI Inicial (com Mido) ---
def is_initial_midi_valid(file_stream):
    try:
        file_stream.seek(0)
        MidoMidiFile(file=file_stream)
        file_stream.seek(0)
        return True
    except Exception:
        return False

# --- Função de Análise com music21 ---
def analyze_midi_with_music21(file_path):
    results = {
        "bpm": "N/A",
        "key": "N/A",
        "time_signature": "N/A",
        "num_bars": "N/A",
        "melodic_range": "N/A",
        "chord_complexity": "N/A",
        "rhythmic_density": "N/A",
        "form_structure": "Ainda não implementado", # Análise de forma é complexa
        "harmonic_progression_preview": "N/A",
        "rhythmic_pattern_summary": "N/A",
        "ai_analysis_text": "Aguardando dados da análise..." # Placeholder
    }

    try:
        s = converter.parse(file_path)
        if not s:
            results["ai_analysis_text"] = "Não foi possível carregar o arquivo com music21."
            return results

        # 1. BPM (Médio)
        bpm_values = []
        for el in s.flat.getElementsByClass(tempo.MetronomeMark):
            bpm_values.append(el.number)
        if bpm_values:
            results["bpm"] = round(sum(bpm_values) / len(bpm_values)) if len(bpm_values) > 1 else round(bpm_values[0])
        elif s.duration: # Fallback se não houver marcas de metrônomo explícitas
            # Tenta estimar a partir de eventos de nota (pode não ser preciso)
            # Para simplificar, se não houver marca explícita, deixamos N/A ou um valor padrão.
            # O ideal é que o MIDI tenha a marcação.
            pass


        # 2. Tom (Key Signature)
        key_obj = s.analyze('key')
        if key_obj:
            # Traduzindo 'major' e 'minor'
            mode_pt = "Maior" if key_obj.mode == 'major' else "Menor" if key_obj.mode == 'minor' else key_obj.mode
            results["key"] = f"{key_obj.tonic.name.replace('-', '♭').replace('#', '♯')} {mode_pt}"

        # 3. Compasso (Time Signature)
        ts_search = s.flat.getElementsByClass(meter.TimeSignature)
        if ts_search:
            results["time_signature"] = ts_search[0].ratioString
        
        # 4. Número de Compassos
        # A forma mais confiável de obter o número do último compasso
        last_measure_number = 0
        for p in s.parts:
            measures_in_part = p.getElementsByClass(stream.Measure)
            if measures_in_part:
                # Alguns MIDIs podem não ter números de compasso explícitos, ou começar do 0.
                # Consideramos o número de objetos Measure como fallback se não houver .number
                m_numbers = [m.number for m in measures_in_part if m.number is not None]
                if m_numbers:
                     last_measure_number = max(last_measure_number, max(m_numbers))
                elif measures_in_part: # Se não houver .number, contamos os objetos
                    last_measure_number = max(last_measure_number, len(measures_in_part))


        if last_measure_number > 0:
             results["num_bars"] = last_measure_number
        elif s.highestTime and results["time_signature"] != "N/A": # Estimativa baseada na duração total
            try:
                ts_obj = meter.TimeSignature(results["time_signature"])
                measure_duration_ql = ts_obj.barDuration.quarterLength
                if measure_duration_ql > 0:
                    results["num_bars"] = int(round(s.highestTime / measure_duration_ql))
            except Exception:
                pass # Mantém N/A se a estimativa falhar

        # 5. Extensão Melódica
        all_pitches = s.flat.pitches # Lista de objetos Pitch
        if all_pitches:
            min_pitch_val = min(p.ps for p in all_pitches)
            max_pitch_val = max(p.ps for p in all_pitches)
            # min_pitch_name = pitch.Pitch(min_pitch_val).nameWithOctave
            # max_pitch_name = pitch.Pitch(max_pitch_val).nameWithOctave
            
            octave_span = (max_pitch_val - min_pitch_val) / 12.0
            if min_pitch_val == max_pitch_val:
                results["melodic_range"] = "Tom único"
            elif octave_span < 0.8: # Menos que uma oitava completa
                results["melodic_range"] = "~ 1 Oitava"
            elif octave_span < 1.5:
                results["melodic_range"] = "1 Oitava"
            elif octave_span < 2.5:
                results["melodic_range"] = "2 Oitavas" # Como no exemplo da imagem
            elif octave_span < 3.5:
                results["melodic_range"] = "3 Oitavas"
            else:
                results["melodic_range"] = f"~ {round(octave_span)} Oitavas"
        
        # 6. Complexidade Harmônica e Progressão
        chordified_stream = s.chordify()
        chords_list = chordified_stream.flat.getElementsByClass(chord.Chord)
        if chords_list:
            # Complexidade (baseado na variedade de qualidades)
            chord_qualities = list(set(c.quality for c in chords_list)) # set para únicos
            if len(chord_qualities) == 0 and len(chords_list) > 0 : # Apenas notas uníssonas ou sem harmonia clara
                results["chord_complexity"] = "Mínima"
            elif len(chord_qualities) <= 2: # ex: Apenas Major e minor
                results["chord_complexity"] = "Simples"
            elif len(chord_qualities) <= 4:
                results["chord_complexity"] = "Moderada" # Como no exemplo
            else:
                results["chord_complexity"] = "Complexa"

            # Preview da Progressão Harmônica (primeiros ~4 acordes diferentes)
            prog_preview_roman = []
            prog_preview_common = []
            # key_for_roman = s.analyze('key') # Já temos key_obj
            
            # Pega os primeiros acordes distintos para o preview
            distinct_chords_for_preview = []
            seen_chord_figures = set()
            for ch in chords_list:
                # Simplifica a figura do acorde para verificar duplicidade no preview
                # (ex: Cmaj7 e Cmaj9 são ambos "C major" na raiz)
                # Para um preview mais útil, podemos pegar os primeiros acordes como aparecem
                if len(distinct_chords_for_preview) < 4:
                     distinct_chords_for_preview.append(ch)
                else:
                    break
            
            for ch_preview in distinct_chords_for_preview:
                try:
                    rn = roman.romanNumeralFromChord(ch_preview, key_obj)
                    prog_preview_roman.append(rn.figureAndKey()) # ex: I in C Major
                except Exception: # Se não conseguir converter para romano
                    prog_preview_roman.append(ch_preview.pitchedCommonName.replace('-', '♭').replace('#', '♯'))
                prog_preview_common.append(ch_preview.pitchedCommonName.replace('-', '♭').replace('#', '♯'))

            if prog_preview_roman:
                results["harmonic_progression_preview"] = " -> ".join(prog_preview_roman)
            elif prog_preview_common :
                 results["harmonic_progression_preview"] = " -> ".join(prog_preview_common)


        # 7. Densidade Rítmica
        notes_and_rests_count = len(s.flat.notesAndRests)
        num_bars_for_density = results["num_bars"]
        if isinstance(num_bars_for_density, int) and num_bars_for_density > 0 and notes_and_rests_count > 0:
            elements_per_measure = notes_and_rests_count / num_bars_for_density
            if elements_per_measure < 8: # Ajustar limiares conforme necessário
                results["rhythmic_density"] = "Baixa"
            elif elements_per_measure < 20:
                results["rhythmic_density"] = "Média" # Como no exemplo
            else:
                results["rhythmic_density"] = "Alta"
        
        # 8. Padrão Rítmico (Sumário - duração mais comum)
        note_durations_ql = [n.duration.quarterLength for n in s.flat.notes]
        if note_durations_ql:
            common_durations = Counter(note_durations_ql).most_common(1)
            if common_durations:
                most_common_ql = common_durations[0][0]
                # Tenta nomear a duração
                try:
                    d_obj = m21duration.Duration(most_common_ql)
                    results["rhythmic_pattern_summary"] = f"Predominância de {d_obj.type}s"
                except Exception:
                    results["rhythmic_pattern_summary"] = f"Duração QL mais comum: {most_common_ql}"
        
        # 9. Texto da "AI Analysis" (construído com os dados)
        analysis_parts = []
        if results["key"] != "N/A":
            analysis_parts.append(f"A tonalidade principal parece ser {results['key']}.")
        if results["bpm"] != "N/A":
            analysis_parts.append(f"O andamento médio é de aproximadamente {results['bpm']} BPM.")
        if results["time_signature"] != "N/A":
            analysis_parts.append(f"Utiliza um compasso de {results['time_signature']}.")
        if results["harmonic_progression_preview"] != "N/A" and results["harmonic_progression_preview"]:
            analysis_parts.append(f"A progressão harmônica inicial observada é: {results['harmonic_progression_preview']}.")
        if results["rhythmic_pattern_summary"] != "N/A":
             analysis_parts.append(f"{results['rhythmic_pattern_summary']}.")
        if results["melodic_range"] != "N/A" and "Oitava" in results["melodic_range"]:
             analysis_parts.append(f"A melodia se estende por {results['melodic_range'].lower()}.")


        if analysis_parts:
            results["ai_analysis_text"] = " ".join(analysis_parts)
        else:
            results["ai_analysis_text"] = "Não foi possível extrair informações detalhadas para a análise textual."


    except Exception as e:
        app.logger.error(f"Erro na análise com music21: {e}")
        results["ai_analysis_text"] = f"Erro ao processar o arquivo MIDI com music21: {str(e)}"
        # Retorna os defaults ou o que foi preenchido até o erro
    
    return results


@app.route('/')
def home():
    # ... (código da rota home existente, sem alterações)
    page_data = {
        "logo_title": "Dark Music Analyzer",
        "intro_title": "Silksong Composition Helper",
        "intro_description": "Importe seus arquivos MIDI, obtenha análises com IA e receba inspiração para continuar seu processo criativo.",
    }
    # Dados mockados para a estrutura da página, serão substituídos no frontend
    analysis_results_data = {"bpm": "...", "key": "...", "bars": "..."}
    ai_analysis_text = "Importe um arquivo MIDI para ver a análise."
    composition_stats_data = [
        {"label": "Extensão Melódica:", "value": "..."},
        {"label": "Complexidade Harmônica:", "value": "..."},
        {"label": "Densidade Rítmica:", "value": "..."},
        {"label": "Estrutura Formal:", "value": "..."} # Manter o original da imagem
    ]
    ai_inspirations_data = [ # Manter como estava
        {"title": "Variação Melódica", "visualizer_bars": [{"width": "70%", "bg_color": "#a78bfa", "margin_top": "0px"}, {"width": "50%", "bg_color": "#a78bfa", "margin_top": "2px"}]},
        {"title": "Extensão Harmônica", "visualizer_bars": [{"width": "60%", "bg_color": "#a78bfa", "margin_top": "0px"}, {"width": "80%", "bg_color": "#a78bfa", "margin_top": "2px"}]},
        {"title": "Desenvolvimento Rítmico", "visualizer_bars": [{"width": "40%", "bg_color": "#a78bfa", "margin_top": "0px"}, {"width": "90%", "bg_color": "#a78bfa", "margin_top": "2px"}]}
    ]
    return render_template(
        'index.html',
        page_data=page_data,
        analysis_results=analysis_results_data, # Placeholders
        ai_analysis=ai_analysis_text, # Placeholder
        composition_stats=composition_stats_data, # Placeholders
        ai_inspirations=ai_inspirations_data
    )

@app.route('/upload_midi', methods=['POST'])
def upload_midi_file():
    if 'midi_file' not in request.files:
        return jsonify({"status": "error", "message": "Nenhum arquivo enviado."}), 400
    file = request.files['midi_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nenhum arquivo selecionado."}), 400

    if file:
        # Validação inicial (extensão e mido)
        allowed_extensions = {'.mid', '.midi'}
        _filename_original, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in allowed_extensions:
            return jsonify({"status": "error", "message": f"Extensão inválida. Use {', '.join(allowed_extensions)}"}), 400
        
        if not is_initial_midi_valid(file.stream): # Validação rápida com Mido
             return jsonify({"status": "error", "filename": file.filename, "message": "Arquivo não parece ser um MIDI válido (validação inicial)."}), 400
        
        # Salvar em arquivo temporário para music21 processar de forma confiável
        temp_file_path = None
        analysis_data = {}
        try:
            # Resetar o ponteiro do stream antes de salvá-lo, pois is_initial_midi_valid já o leu.
            file.stream.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                # FileStorage.save() espera um dst (destination path or stream)
                # Para salvar o conteúdo do stream do FileStorage:
                tmp.write(file.stream.read())
                temp_file_path = tmp.name
            
            analysis_data = analyze_midi_with_music21(temp_file_path)
            
            return jsonify({
                "status": "success",
                "filename": file.filename,
                "message": "Análise MIDI concluída.",
                "analysis": analysis_data # Adiciona os resultados da análise
            }), 200

        except Exception as e:
            app.logger.error(f"Erro geral no upload ou análise: {e}")
            return jsonify({"status": "error", "filename": file.filename, "message": f"Erro no processamento do arquivo: {str(e)}"}), 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path) # Limpa o arquivo temporário
    
    return jsonify({"status": "error", "message": "Falha no upload."}), 500


if __name__ == '__main__':
    us = environment.UserSettings()
    us['warnings'] = 0
    app.run(debug=True)
