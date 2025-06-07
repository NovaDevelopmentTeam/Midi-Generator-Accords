import pretty_midi
import numpy as np

VOCAB_SIZE = 129  # 0-127: Noten, 128: Pause
PAUSE_TOKEN = 128
CHORD_TOKEN = 129  # Neues Token für Akkorde

def encode_midi(pm: pretty_midi.PrettyMIDI, time_step=0.125) -> np.ndarray:
    """Konvertiert MIDI in Token-Sequenz mit Akkorden"""
    # Sammle alle Noten aus allen Instrumenten
    all_notes = []
    for instrument in pm.instruments:
        all_notes.extend(instrument.notes)
    
    if not all_notes:
        return np.array([], dtype=np.int32) # int64
    
    # Zeitachse erstellen
    end_time = max(note.end for note in all_notes)
    time_steps = np.arange(0, end_time + time_step, time_step)
    events = []
    
    for t in time_steps:
        # Finde aktive Noten zum Zeitpunkt t
        active_notes = [note.pitch for note in all_notes if note.start <= t < note.end]
        
        if not active_notes:
            events.append(PAUSE_TOKEN)
        elif len(active_notes) == 1:
            events.append(active_notes[0])
        else:
            # Akkord: Noten sortieren für Konsistenz
            events.extend(sorted(active_notes))
            events.append(CHORD_TOKEN)
    
    return np.array(events, dtype=np.int32) # int64

def decode_sequence(seq: np.ndarray, output_path: str, time_step=0.125):
    """Rekonstruiert MIDI aus Token-Sequenz mit Akkorden"""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    time = 0.0
    current_notes = []
    
    for token in seq:
        if token == PAUSE_TOKEN:
            # Pause: Zeit vorrücken
            time += time_step
        elif token == CHORD_TOKEN:
            # Akkord-Ende: Noten hinzufügen
            for pitch in current_notes:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(pitch),
                    start=time,
                    end=time + time_step
                )
                piano.notes.append(note)
            current_notes = []
            time += time_step
        elif 0 <= token <= 127:
            # Einzelnote oder Akkord-Noten
            current_notes.append(token)
        else:
            # Ungültiges Token überspringen
            continue
    
    pm.instruments.append(piano)
    pm.write(output_path)