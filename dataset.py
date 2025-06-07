import os
import pretty_midi
import torch
from torch.utils.data import Dataset
from utils import encode_midi
from config import Config

cfg = Config()

class MIDIDataset(Dataset):
    def __init__(self, midi_folder: str, seq_length: int):
        self.seq_length = cfg.seq_length # self.seq_length = seq_length
        self.files = [os.path.join(midi_folder, f) 
                     for f in os.listdir(midi_folder) 
                     if f.endswith(('.mid', '.midi'))]
        self.sequences = []

        for path in self.files:
            try:
                pm = pretty_midi.PrettyMIDI(path)
                tokens = encode_midi(pm)
                self.sequences.append(tokens)
                print(f"Loaded: {path} ({len(tokens)} tokens)")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
    
    def __len__(self):
        return sum(len(seq) - self.seq_length for seq in self.sequences)
    
    def __getitem__(self, idx):
        cumulative = 0
        for seq in self.sequences:
            if idx < cumulative + len(seq) - self.seq_length:
                start_idx = idx - cumulative
                end_idx = start_idx + self.seq_length
                x = seq[start_idx:end_idx]
                y = seq[start_idx + 1:end_idx + 1]
                return torch.tensor(x), torch.tensor(y)
            cumulative += len(seq) - self.seq_length
        raise IndexError("Index out of range")