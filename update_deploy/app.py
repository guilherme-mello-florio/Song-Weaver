from flask import Flask, render_template, request, jsonify, url_for
import random
import os
import tempfile
import google.generativeai as genai
import json
import re
import hashlib
import copy
import statistics 
import math

from music21 import converter, tempo, pitch, key, environment, stream, note, chord, roman, common, meter, duration as m21duration
from collections import Counter
from music21 import analysis as m21analysis

from mido import MidiFile as MidoMidiFile

# Conexão utilizando key da API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Erro ao configurar a API de geração: Verifique sua GEMINI_API_KEY. Erro: {e}")

app = Flask(__name__)

# Cache para armazenar os resultados das gerações de MIDI
MIDI_GENERATION_CACHE = {}


def separate_piano_parts(s):
    """
    Separa uma stream music21 em partes de mão direita (aguda) e mão esquerda (grave).
    Retorna (rh_part, lh_part). lh_part pode ser None se for uma stream de parte única.
    """
    # Se o arquivo já possui partes, tenta usá-las
    if len(s.parts) > 1:
        # Um caso comum para MIDI de piano são duas partes
        part1 = s.parts[0]
        part2 = s.parts[1]

        avg_pitch1 = 0
        avg_pitch2 = 0
        
        notes1 = list(part1.flatten().pitches)
        notes2 = list(part2.flatten().pitches)

        if not notes1 and not notes2: return (None, None)
        if not notes1: return (part2, None)
        if not notes2: return (part1, None)

        avg_pitch1 = sum(p.ps for p in notes1) / len(notes1)
        avg_pitch2 = sum(p.ps for p in notes2) / len(notes2)
        
        # Compara as alturas médias para definir mão direita (RH) e esquerda (LH)
        if avg_pitch1 > avg_pitch2:
            return (part1, part2) # part1 é RH, part2 é LH
        else:
            return (part2, part1) # part2 é RH, part1 é LH
            
    # Se for uma stream 'flat' (sem partes), temos que criar as partes dividindo o alcance das notas
    else:
        # Por enquanto, se não houver partes, trata como uma única parte (mão direita).
        return (s, None)


def midi_stream_to_text(s, limit=64):
    """
    Converte uma stream music21 (ou parte) em uma representação de texto (JSON).
    O limite aumentado fornece mais contexto para a geração.
    """
    if not s:
        return "[]" # Retorna um array JSON vazio se a parte for None
        
    components = []
    # Itera sobre os últimos 'limit' elementos da stream
    for element in s.flatten().notesAndRests[-limit:]:
        velocity = 80 # Velocity padrão
        if hasattr(element, 'volume') and element.volume.velocity is not None:
            velocity = element.volume.velocity

        if isinstance(element, note.Note):
            components.append({
                "type": "note",
                "pitch": element.pitch.nameWithOctave,
                "offset": float(element.offset),
                "quarterLength": float(element.duration.quarterLength),
                "velocity": velocity
            })
        elif isinstance(element, chord.Chord):
            # Calcula a velocidade média para acordes
            avg_velocity = round(sum(n.volume.velocity for n in element if n.volume.velocity is not None) / len(element)) if len(element) > 0 else 80
            components.append({
                "type": "chord",
                "pitches": [p.nameWithOctave for p in element.pitches],
                "offset": float(element.offset),
                "quarterLength": float(element.duration.quarterLength),
                "velocity": avg_velocity
            })
        elif isinstance(element, note.Rest):
            components.append({
                "type": "rest",
                "offset": float(element.offset),
                "quarterLength": float(element.duration.quarterLength),
            })
    return json.dumps(components, indent=2)

def generate_music_continuation_with_gemini(analysis_data, music_text_rh, music_text_lh):
    """
    Gera a continuação da música usando o modelo generativo.
    """
    
    # Define o modelo generativo
    model = genai.GenerativeModel('models/gemini-pro-latest')

    # Instruções para a geração da continuação
    prompt = f"""
    Você é um compositor especialista em piano, mestre em contraponto, harmonia e desenvolvimento estilístico. Sua tarefa é compor uma continuação para uma peça de piano de duas mãos.

    # ANÁLISE GERAL DA MÚSICA #
    - Tonalidade: {analysis_data.get('key', 'N/A')}
    - Andamento (BPM): {analysis_data.get('bpm', 'N/A')}
    - Compasso: {analysis_data.get('time_signature', 'N/A')}
    - Último offset (tempo final): {analysis_data.get('last_offset', 0.0)}

    # MÃO DIREITA (Melodia/Harmonia Superior) - ÚLTIMOS COMPASSOS #
    ```json
    {music_text_rh}
    ```

    # MÃO ESQUERDA (Baixo/Acompanhamento) - ÚLTIMOS COMPASSOS #
    (Se for `[]` ou `null`, significa que a peça original tinha apenas uma linha, e você deve criar um acompanhamento para a mão esquerda.)
    ```json
    {music_text_lh}
    ```

    # SUA TAREFA: COMPOR UMA CONTINUAÇÃO PARA AMBAS AS MÃOS #
    Crie uma continuação de 4 a 8 compassos que se integre perfeitamente. A continuação deve ser uma frase de desenvolvimento, não uma conclusão.

    1.  **FUNÇÃO DAS MÃOS:** Mantenha a textura original.
        -   **Mão Direita:** Continue as ideias melódicas ou os padrões de acordes da parte original.
        -   **Mão Esquerda:** Forneça suporte harmônico e rítmico. Continue o padrão de acompanhamento (ex: baixo de Alberti, acordes quebrados, linha de baixo). Se a mão esquerda original não foi fornecida, crie um acompanhamento apropriado para a mão direita.

    2.  **INTERAÇÃO E COERÊNCIA:** As duas mãos devem soar como se pertencessem à mesma peça. Elas devem se complementar ritmica e harmonicamente.

    3.  **HARMONIA DE DESENVOLVIMENTO (REGRA CRÍTICA):**
        -   **NÃO TERMINE NA TÔNICA (I).** Sua continuação deve terminar em um acorde que cria expectativa, como o acorde de **Dominante (V)**, para que o compositor se sinta inspirado a continuar.

    4.  **EXPRESSÃO E DINÂMICA (VELOCITY):** Varie a `velocity` em ambas as mãos para criar um fraseado musical expressivo.

    5.  **PONTO DE PARTIDA (OFFSET):** O primeiro evento em ambas as mãos deve começar no ou após o "Último offset" fornecido.

    # FORMATO DA RESPOSTA #
    Responda APENAS com um único objeto JSON contendo duas chaves: "right_hand" e "left_hand". Cada chave deve conter uma lista de objetos de nota/acorde/pausa. O JSON deve começar estritamente com `{{` e terminar com `}}`.

    Exemplo de formato de resposta:
    {{
      "right_hand": [ {{ "type": "note", ... }} ],
      "left_hand": [ {{ "type": "chord", ... }} ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        text_response = response.text
        
        # Extração robusta de JSON para lidar com markdown e texto conversacional.
        # Primeiro, tenta encontrar um bloco JSON dentro de cercas de markdown
        match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
        if match:
            return match.group(1) # Retorna apenas o conteúdo dentro das cercas

        # Se não houver cercas de markdown, procura o primeiro objeto JSON bruto
        match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if match:
            return match.group(0)

        # Se ainda assim não houver correspondência, registra a falha e retorna None
        app.logger.error("Nenhum JSON válido (com ou sem markdown) encontrado na resposta.")
        app.logger.error(f"Resposta recebida: {text_response}")
        return None
        
    except Exception as e:
        app.logger.error(f"Erro ao chamar a API de geração: {e}")
        return None

def text_to_midi_stream(text_data, original_bpm=120):
    """Converte a representação JSON de texto de volta para um stream do music21."""
    new_stream = stream.Part() # Gera uma stream 'Part' (Parte), não uma Stream geral
    
    try:
        if not text_data or text_data.strip() == "":
            return new_stream # Retorna uma parte vazia se não houver dados

        music_elements = json.loads(text_data)
        if not music_elements: 
            return new_stream # Retorna uma parte vazia se o JSON estiver vazio

        for element_data in music_elements:
            offset = float(element_data.get("offset", 0.0))
            duration = element_data.get("quarterLength")
            velocity = int(element_data.get("velocity", 80))

            if element_data["type"] == "note":
                new_note = note.Note(element_data["pitch"])
                new_note.duration.quarterLength = float(duration)
                new_note.volume.velocity = velocity
                new_stream.insert(offset, new_note)
            elif element_data["type"] == "chord":
                new_chord = chord.Chord(element_data["pitches"])
                new_chord.duration.quarterLength = float(duration)
                for n_in_chord in new_chord:
                    n_in_chord.volume.velocity = velocity
                new_stream.insert(offset, new_chord)
            elif element_data["type"] == "rest":
                new_rest = note.Rest()
                new_rest.duration.quarterLength = float(duration)
                new_stream.insert(offset, new_rest)

        return new_stream
    
    except json.JSONDecodeError as e:
        app.logger.error(f"Erro de decodificação de JSON ao converter texto para stream MIDI: {e}")
        app.logger.error(f"--- Resposta (JSON) que causou o erro\n{text_data}\n------------------------------------")
        return None # Crítico: retorna None em caso de falha
    except Exception as e:
        app.logger.error(f"Erro ao converter texto para stream MIDI: {e}")
        return None # Crítico: retorna None em caso de falha

def is_initial_midi_valid(file_stream):
    """Verifica se o stream do arquivo é um MIDI válido usando o Mido."""
    try:
        file_stream.seek(0)
        MidoMidiFile(file=file_stream) # Tenta carregar o arquivo com Mido
        file_stream.seek(0) # Rebobina o stream para uso posterior
        return True
    except Exception:
        return False # Falha ao carregar

def humanize_stream(music_stream):
    """Aplica micro-variações de tempo e dinâmica para um toque mais humano."""
    for n in music_stream.flatten().notes:
        # Variação sutil de tempo
        timing_variation = random.uniform(-0.0075, 0.0075)
        n.offset += timing_variation
        
        # Variação sutil de velocidade
        velocity_variation = random.randint(-4, 4)
        new_velocity = n.volume.velocity + velocity_variation
        n.volume.velocity = max(0, min(127, new_velocity)) # Garante que fique entre 0 e 127
    return music_stream

def analyze_midi_with_music21(file_path):
    """
    Função de análise de MIDI robusta, com tratamento de exceções 
    e validação de sanidade para BPM, Compasso e Tonalidade.
    """
    results = {
        "bpm": "N/A", "key": "N/A", "time_signature": "N/A", "num_bars": "N/A",
        "melodic_range": "N/A", "chord_complexity": "N/A", "rhythmic_density": "N/A",
        "form_structure": "Ainda não implementado", "harmonic_progression_preview": "N/A",
        "rhythmic_pattern_summary": "N/A", "ai_analysis_text": "Aguardando dados da análise..."
    }
    s = None
    
    try:
        s = converter.parse(file_path)
        if not s:
            results["ai_analysis_text"] = "Não foi possível carregar o arquivo com music21."
            return results, s 
        
        # TRATAMENTO DE EXCEÇÃO: MIDI VAZIO (sem notas) 
        if not s.flat.notesAndRests:
            results["ai_analysis_text"] = "O arquivo MIDI foi carregado, mas não contém notas ou pausas."
            return results, s

        # TRATAMENTO DE EXCEÇÃO: BPM 
        bpm_values = []
        MIN_BPM = 30  # Limite mínimo razoável
        MAX_BPM = 280 # Limite máximo razoável

        for el in s.flat.getElementsByClass(tempo.MetronomeMark):
            if MIN_BPM <= el.number <= MAX_BPM:
                bpm_values.append(el.number)

        if bpm_values:
            # Usa a MEDIANA (resistente a outliers)
            results["bpm"] = round(statistics.median(bpm_values))
        else:
            try:
                # Tenta estimar o tempo se não houver marcação explícita
                estimated_tempo = s.estimateTempo()
                if estimated_tempo:
                    # "Prende" (clamp) o valor estimado dentro dos limites
                    results["bpm"] = round(max(MIN_BPM, min(MAX_BPM, estimated_tempo.number)))
                else:
                    results["bpm"] = 120 # Fallback
            except Exception:
                results["bpm"] = 120 # Fallback final
        
        # TRATAMENTO DE EXCEÇÃO: Compasso (Inferência e Fallback)
        ts_obj = None
        ts_search = s.flat.getElementsByClass(meter.TimeSignature)
        suspicious_ts_flag = False
        original_ts_str = ""
        
        if ts_search:
            ts_obj = ts_search[0]
        else:
            try:
                # 1. Tenta INFERIR o compasso
                app.logger.info("Compasso não encontrado. Tentando inferir...")
                ts_obj = meter.bestTimeSignature(s)
                app.logger.info(f"Compasso inferido: {ts_obj.ratioString}")
            except Exception as e:
                # 2. Se a inferência falhar, usa um FALLBACK
                app.logger.warning(f"Falha ao inferir compasso ({e}). Usando 4/4.")
                ts_obj = meter.TimeSignature('4/4') # Padrão mais comum
            
           # 3. Insere o compasso (inferido ou padrão) na stream
            s.insert(0, ts_obj)

        # NOVO TRATAMENTO: Simplificação de Compasso (Ex: 12/16 -> 3/4) 
        # Adicionado para tratar compassos com numeradores/denominadores > 8
        try:
            num = ts_obj.numerator
            den = ts_obj.denominator
            
            # Se o numerador OU denominador for maior que 8, tenta simplificar
            if num > 8 or den > 8:
                common_divisor = math.gcd(num, den)
                if common_divisor > 1:
                    new_num = num // common_divisor
                    new_den = den // common_divisor
                    simplified_ts_str = f'{new_num}/{new_den}'
                    app.logger.info(f"Simplificando compasso de {ts_obj.ratioString} para {simplified_ts_str}")
                    ts_obj = meter.TimeSignature(simplified_ts_str)
                    
                    # Remove o compasso antigo e insere o novo simplificado
                    s.removeByClass(meter.TimeSignature)
                    s.insert(0, ts_obj)
        except Exception as e:
            app.logger.warning(f"Falha ao tentar simplificar o compasso: {e}")
        # FIM DA SIMPLIFICAÇÃO


        # NOVO TRATAMENTO: Compassos "Suspeitos" (1/4, 2/4)
        original_ts_str = ts_obj.ratioString # Guarda o compasso original (ex: "3/4")

        if original_ts_str in ['1/4', '2/4']:
            app.logger.warning(f"Compasso musicalmente incomum detectado: {original_ts_str}. Usando 4/4 como padrão de análise.")
            suspicious_ts_flag = True
            ts_obj = meter.TimeSignature('4/4') # Define um padrão mais seguro
            s.insert(0, ts_obj) # Sobrescreve o compasso na stream para os cálculos
        
        results["time_signature"] = ts_obj.ratioString # Armazena o compasso SEGURO (ex: 4/4)

        # TRATAMENTO DE EXCEÇÃO: Tonalidade (Validação de Confiança)
        key_obj = s.analyze('key')
        
        # Verifica se a análise de tonalidade é confiável
        if key_obj and key_obj.correlationCoefficient > 0.70:
            mode_pt = "Maior" if key_obj.mode == 'major' else "Menor" if key_obj.mode == 'minor' else key_obj.mode
            results["key"] = f"{key_obj.tonic.name.replace('-', '♭').replace('#', '♯')} {mode_pt}"
        else:
            # Se a confiança for baixa (música atonal, curta, etc.), não afirma a tonalidade
            results["key"] = "Indefinido"
            key_obj = None # Anula para que a análise de Graus Romanos não seja usada

        # INÍCIO DA ANÁLISE DETALHADA

        if s.highestTime:
            results["last_offset"] = float(s.highestTime) # Offset final da música

        chord_stream_list = list(s.chordify().flat.getElementsByClass(chord.Chord))
        last_chord = chord_stream_list[-1] if chord_stream_list else None
        
        if last_chord:
            # Só tenta análise de Graus Romanos se a tonalidade for confiável
            if key_obj:
                try:
                    rn = roman.romanNumeralFromChord(last_chord, key_obj)
                    results["final_chord_analysis"] = f"A música termina em um acorde {last_chord.pitchedCommonName}, que funciona como um grau {rn.figure}."
                except Exception:
                    results["final_chord_analysis"] = f"A música termina no acorde {last_chord.pitchedCommonName}."
            else:
                results["final_chord_analysis"] = f"A música termina no acorde {last_chord.pitchedCommonName}."
            
            last_melodic_note = last_chord.pitches[-1] # Nota mais aguda do último acorde
            if key_obj:
                scale_degree = key_obj.getScaleDegreeFromPitch(last_melodic_note)
                degree_names = ["Tônica", "Supertônica", "Mediante", "Subdominante", "Dominante", "Superdominante", "Sensível"]
                if scale_degree and 1 <= scale_degree <= 7:
                    degree_name = degree_names[scale_degree-1]
                    results["final_melody_analysis"] = f"A melodia termina na nota {last_melodic_note.name} ({degree_name})."
                else:
                    results["final_melody_analysis"] = f"A melodia termina na nota {last_melodic_note.name}."
        
        # Cálculo de número de compassos
        last_measure_number = 0
        for p in s.parts:
            measures_in_part = p.getElementsByClass(stream.Measure)
            if measures_in_part:
                m_numbers = [m.number for m in measures_in_part if m.number is not None]
                if m_numbers:
                    last_measure_number = max(last_measure_number, max(m_numbers))
                elif measures_in_part:
                    # Fallback para partes sem números de compasso explícitos
                    last_measure_number = max(last_measure_number, len(measures_in_part))
        
        if last_measure_number > 0:
            results["num_bars"] = last_measure_number
        elif s.highestTime and ts_obj: # Fallback
            try:
                measure_duration_ql = ts_obj.barDuration.quarterLength
                if measure_duration_ql > 0:
                    results["num_bars"] = int(round(s.highestTime / measure_duration_ql))
            except Exception:
                pass # num_bars permanece "N/A"

        # Análise da extensão melódica
        all_pitches = s.flat.pitches
        if all_pitches:
            min_pitch_val = min(p.ps for p in all_pitches)
            max_pitch_val = max(p.ps for p in all_pitches)
            octave_span = (max_pitch_val - min_pitch_val) / 12.0
            if octave_span < 1.5: results["melodic_range"] = "1-2 Oitavas"
            elif octave_span < 3: results["melodic_range"] = "2-3 Oitavas"
            else: results["melodic_range"] = f"~ {round(octave_span)} Oitavas"
        
        # Análise de complexidade harmônica
        if chord_stream_list:
            chord_qualities = list(set(c.quality for c in chord_stream_list))
            if len(chord_qualities) <= 2: results["chord_complexity"] = "Simples"
            elif len(chord_qualities) <= 4: results["chord_complexity"] = "Moderada"
            else: results["chord_complexity"] = "Complexa"
            
            
            prog_preview_roman = []
            
            
            distinct_chords_for_preview = []
            seen_chord_names = set() # Usamos um set para rastrear nomes (ex: "C major triad")

            for ch in chord_stream_list:
                if len(distinct_chords_for_preview) >= 4:
                    break # Já temos 4 acordes para a prévia

                current_chord_name = ch.pitchedCommonName
                if current_chord_name not in seen_chord_names:
                    # Se for um nome de acorde que ainda não vimos, adiciona
                    distinct_chords_for_preview.append(ch)
                    seen_chord_names.add(current_chord_name)
            

            for ch_preview in distinct_chords_for_preview:
                if key_obj:
                    try:
                        # Tenta obter o grau romano
                        rn = roman.romanNumeralFromChord(ch_preview, key_obj)
                        prog_preview_roman.append(rn.figure)
                    except Exception:
                        prog_preview_roman.append(ch_preview.pitchedCommonName.replace('-', '♭').replace('#', '♯'))
                else:
                    # Se a tonalidade é indefinida, usa o nome do acorde
                    prog_preview_roman.append(ch_preview.pitchedCommonName.replace('-', '♭').replace('#', '♯'))
            
            if prog_preview_roman:
                results["harmonic_progression_preview"] = " -> ".join(prog_preview_roman)

        # Análise de densidade rítmica
        notes_and_rests_count = len(s.flat.notesAndRests)
        num_bars_for_density = results["num_bars"]
        if isinstance(num_bars_for_density, int) and num_bars_for_density > 0:
            elements_per_measure = notes_and_rests_count / num_bars_for_density
            if elements_per_measure < 8: results["rhythmic_density"] = "Baixa"
            elif elements_per_measure < 20: results["rhythmic_density"] = "Média"
            else: results["rhythmic_density"] = "Alta"
        
        # Análise de padrões rítmicos (duração mais comum)
        note_durations_ql = [n.duration.quarterLength for n in s.flat.notes]
        if note_durations_ql:
            common_durations = Counter(note_durations_ql).most_common(1)
            if common_durations:
                most_common_ql = common_durations[0][0]
                try:
                    d_obj = m21duration.Duration(most_common_ql)
                    results["rhythmic_pattern_summary"] = f"Predominância de {d_obj.type}s"
                except Exception:
                    results["rhythmic_pattern_summary"] = f"Duração QL: {most_common_ql}"

        # Geração do texto de análise
        analysis_parts = []
        if results["key"] != "N/A": analysis_parts.append(f"A tonalidade principal parece ser {results['key']}.")
        if results["bpm"] != "N/A": analysis_parts.append(f"O andamento médio é de aproximadamente {results['bpm']} BPM.")
        if suspicious_ts_flag:
            # Caso o compasso seja muito distoante e sem sentido
            analysis_parts.append(f"Detectado compasso de {original_ts_str}. A estrutura rítmica é provavelmente 3/4 ou 4/4.")
        elif results["time_signature"] != "N/A":
            # Caso contrário, usa o compasso normal
            analysis_parts.append(f"Utiliza um compasso de {results['time_signature']}.")
        if results["harmonic_progression_preview"] != "N/A" and results["harmonic_progression_preview"]: analysis_parts.append(f"A progressão harmônica inicial observada é: {results['harmonic_progression_preview']}.")
        if results["rhythmic_pattern_summary"] != "N/A": analysis_parts.append(f"{results['rhythmic_pattern_summary']}.")
        if results["melodic_range"] != "N/A": analysis_parts.append(f"A melodia se estende por {results['melodic_range'].lower()}.")
        if analysis_parts:
            results["ai_analysis_text"] = " ".join(analysis_parts)
        else:
            results["ai_analysis_text"] = "Não foi possível extrair informações detalhadas."
    
        # FIM DA ANÁLISE DETALHADA

    except Exception as e:
        app.logger.error(f"Erro na análise com music21: {e}")
        results["ai_analysis_text"] = f"Erro ao processar o arquivo MIDI: {str(e)}"
    
    return results, s


@app.route('/')
def home():
    """Renderiza a página inicial (index.html)."""
    
    # Dados para preenchimento dinâmico do template
    page_data = {
        "logo_title": "Dark Music Analyzer", "intro_title": "Silksong Composition Helper",
        "intro_description": "Importe seus arquivos MIDI, obtenha análises e receba inspiração para continuar seu processo criativo.",
    }
    analysis_results_data = {"bpm": "...", "key": "...", "bars": "..."}
    ai_analysis_text = "Importe um arquivo MIDI para ver a análise."
    composition_stats_data = [
        {"label": "Extensão Melódica:", "value": "..."}, {"label": "Complexidade Harmônica:", "value": "..."},
        {"label": "Densidade Rítmica:", "value": "..."}, {"label": "Estrutura Formal:", "value": "..."}
    ]
    return render_template('index.html', page_data=page_data, analysis_results=analysis_results_data,
        ai_analysis=ai_analysis_text, composition_stats=composition_stats_data)

@app.route('/upload_midi', methods=['POST'])
def upload_midi_file():
    """Rota para upload, análise e geração de continuação do MIDI."""
    
    if 'midi_file' not in request.files:
        return jsonify({"status": "error", "message": "Nenhum arquivo enviado."}), 400
    file = request.files['midi_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Nenhum arquivo selecionado."}), 400

    if file:
        # TRATAMENTO DE EXCEÇÃO: Validação inicial do Mido
        if not is_initial_midi_valid(file.stream):
             return jsonify({"status": "error", "filename": file.filename, "message": "Arquivo não parece ser um MIDI válido."}), 400

        temp_file_path = None
        try:
            file.stream.seek(0)
            file_content = file.stream.read()
            # Cria um hash do conteúdo do arquivo para usar como chave de cache
            file_hash = hashlib.md5(file_content).hexdigest()

            # Verifica se o resultado já está no cache
            if file_hash in MIDI_GENERATION_CACHE:
                cached_response = MIDI_GENERATION_CACHE[file_hash]
                cached_response['filename'] = file.filename
                return jsonify(cached_response)

            # Salva o conteúdo em um arquivo temporário para análise
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
                tmp.write(file_content)
                temp_file_path = tmp.name
            
            # CHAMADA DA FUNÇÃO ROBUSTA
            # Esta função trata compasso, bpm, tonalidade, etc.
            analysis_data, original_stream = analyze_midi_with_music21(temp_file_path)
            
            # TRATAMENTO DE EXCEÇÃO: Verifica se a análise teve sucesso
            if not original_stream or not original_stream.flat.notesAndRests:
                return jsonify({"status": "error", "message": "Falha ao analisar o arquivo ou arquivo está vazio.", "analysis": analysis_data}), 500

            # Separa as partes e converte para texto (JSON)
            rh_part_orig, lh_part_orig = separate_piano_parts(original_stream)
            music_as_text_rh = midi_stream_to_text(rh_part_orig)
            music_as_text_lh = midi_stream_to_text(lh_part_orig)

            # Gera a continuação
            generated_text = generate_music_continuation_with_gemini(analysis_data, music_as_text_rh, music_as_text_lh)
            
            generated_midi_url = None
            combined_midi_url = None # Esta variável não está sendo usada, mas foi mantida

            if generated_text:
                app.logger.info(f"--- Tentando decodificar o seguinte JSON\n{generated_text}\n------------------------------------")
                generated_parts_json = json.loads(generated_text)
                generated_text_rh = json.dumps(generated_parts_json.get("right_hand", []))
                generated_text_lh = json.dumps(generated_parts_json.get("left_hand", []))

                bpm = analysis_data.get('bpm', 120)
                if not isinstance(bpm, (int, float)): bpm = 120
                
                # Cria duas streams separadas para as partes geradas
                # text_to_midi_stream é robusto e retorna None em caso de falha de JSON
                raw_generated_part_rh = text_to_midi_stream(generated_text_rh, bpm)
                if raw_generated_part_rh is None:
                    app.logger.error("Falha ao converter RH text_to_midi_stream. Criando parte vazia.")
                    raw_generated_part_rh = stream.Part() # Cria uma parte vazia como fallback

                raw_generated_part_lh = text_to_midi_stream(generated_text_lh, bpm)
                if raw_generated_part_lh is None:
                    app.logger.error("Falha ao converter LH text_to_midi_stream. Criando parte vazia.")
                    raw_generated_part_lh = stream.Part() # Cria uma parte vazia como fallback

                # Constrói o arquivo MIDI "somente da continuação"
                continuation_stream = stream.Stream()
                continuation_stream.insert(0, tempo.MetronomeMark(number=bpm)) # Adiciona o BPM

                # Normaliza os offsets para começar do 0 para o player independente
                first_offset_rh = raw_generated_part_rh.flatten().notesAndRests.first().offset if raw_generated_part_rh.flatten().notesAndRests else float('inf')
                first_offset_lh = raw_generated_part_lh.flatten().notesAndRests.first().offset if raw_generated_part_lh.flatten().notesAndRests else float('inf')
                min_first_offset = min(first_offset_rh, first_offset_lh)
                
                # Só faz o shift se houver notas e o offset não for infinito
                if min_first_offset != float('inf') and min_first_offset > 0:
                    raw_generated_part_rh.shiftElements(-min_first_offset)
                    raw_generated_part_lh.shiftElements(-min_first_offset)
                
                # Insere as partes na stream de continuação
                if list(raw_generated_part_rh.flatten().notesAndRests):
                    continuation_stream.insert(0, raw_generated_part_rh)
                if list(raw_generated_part_lh.flatten().notesAndRests):
                    continuation_stream.insert(0, raw_generated_part_lh)
                
                # Salva o arquivo MIDI gerado
                output_dir = os.path.join('static', 'generated')
                os.makedirs(output_dir, exist_ok=True)
                continuation_filename = f"continuation_{file_hash}.mid"
                continuation_filepath = os.path.join(output_dir, continuation_filename)
                continuation_stream.write('midi', fp=continuation_filepath)
                generated_midi_url = url_for('static', filename=f'generated/{continuation_filename}')

            # Prepara a resposta final
            final_response = {
                "status": "success", "filename": file.filename, "message": "Análise e geração concluídas.",
                "analysis": analysis_data, "generated_midi_url": generated_midi_url
            }
            # Armazena a resposta no cache
            MIDI_GENERATION_CACHE[file_hash] = final_response
            return jsonify(final_response), 200

        except json.JSONDecodeError as e:
            app.logger.error(f"Erro Crítico de Decodificação de JSON: {e}")
            app.logger.error(f"Resposta (JSON) que causou o erro: {generated_text}")
            return jsonify({"status": "error", "filename": file.filename, "message": f"Erro ao ler a resposta da geração: {str(e)}"}), 500
        except Exception as e:
            app.logger.error(f"Erro geral no upload ou análise: {e}", exc_info=True)
            return jsonify({"status": "error", "filename": file.filename, "message": f"Erro no processamento: {str(e)}"}), 500
        finally:
            # Garante que o arquivo temporário seja excluído
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    return jsonify({"status": "error", "message": "Falha no upload."}), 500


if __name__ == '__main__':
    # Configurações de ambiente do music21
    us = environment.UserSettings()
    us['warnings'] = 0 # Desativa os avisos do music21
    
    # Inicia o servidor Flask em modo de depuração
    app.run(debug=True)