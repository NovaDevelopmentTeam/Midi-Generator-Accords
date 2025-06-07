import os
import torch
import pretty_midi
from torch.utils.data import Dataset
from config import Config

cfg = Config()
USE_CACHE = True  # Über diese Variable kannst du Caching aktivieren/deaktivieren
CACHE_PATH = "cached_dataset.pt"

class MIDIDataset(Dataset):
    def __init__(self, midi_folder, seq_length=cfg.seq_length): # seq_length=128
        self.seq_length = seq_length

        if USE_CACHE and os.path.exists(CACHE_PATH):
            print("Lade MIDI-Daten aus Cache...")
            self.samples = torch.load(CACHE_PATH)
            return

        self.samples = []
        midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith(".mid") or f.endswith(".midi")]

        for midi_path in midi_files:
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
                events = []

                for instrument in midi.instruments:
                    if instrument.is_drum:
                        continue

                    for chord in self.extract_chords(instrument.notes):
                        events.append(f"CHORD_{'_'.join(str(n) for n in chord)}")

                # Tokenisierung und Fensterung
                tokens = [hash(e) % 32768 for e in events]  # einfache Tokendarstellung
                for i in range(0, len(tokens) - seq_length):
                    self.samples.append(torch.tensor(tokens[i:i + seq_length]))
                    print(f"Loaded: {midi_path} ({len(tokens)} tokens)")

            except Exception as e:
                print(f"Fehler beim Verarbeiten von {midi_path}: {e}")

        if USE_CACHE:
            torch.save(self.samples, CACHE_PATH)

    def extract_chords(self, notes):
        if not notes:
            return []

        # Sortiere nach Startzeit
        notes = sorted(notes, key=lambda n: n.start)
        chords = []
        chord = [notes[0].pitch]
        current_time = notes[0].start

        for note in notes[1:]:
            if abs(note.start - current_time) < 0.05:  # innerhalb 50ms -> gleiches Akkordcluster
                chord.append(note.pitch)
            else:
                chords.append(sorted(chord))
                chord = [note.pitch]
                current_time = note.start

        chords.append(sorted(chord))
        return chords

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
