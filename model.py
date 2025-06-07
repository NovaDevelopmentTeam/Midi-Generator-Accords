import torch
import torch.nn as nn
from config import Config

class MusicLSTM(nn.Module):
    def __init__(self, cfg = Config()): # def __init__(self, cfg):
        super().__init__()
        self.vocab_size = 130  # 0-127: Noten, 128: Pause, 129: Akkord-Token
        self.embed = nn.Embedding(self.vocab_size, cfg.hidden_size)
        self.lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(cfg.hidden_size, self.vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden