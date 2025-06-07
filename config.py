from dataclasses import dataclass
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    midi_folder: str = "data\midi_files" # "C:\Users\jonas\Downloads\midi_gen\v2\midi_ai\new\version_with_accords\data"
    batch_size: int = 16 if device == "cuda" else 8 # 64/32, 32/16
    seq_length: int = 50 if device == "cuda" else 25 # 100/50
    hidden_size: int = 256 # 512
    num_layers: int = 4
    dropout: float = 0.3 # 0.3
    lr: float = 0.001
    epochs: int = 100 if device == "cuda" else 50
    device: str = device
    checkpoint_path: str = "model_checkpoint.pth"
    max_loaded_files = 500  # Höheres Limit für mehr Daten