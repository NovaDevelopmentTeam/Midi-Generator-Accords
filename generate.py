import os
import torch
import numpy as np
import random
from config import Config
from model import MusicLSTM
from utils import decode_sequence
from dataset import MIDIDataset  # für Startsequenz aus Dataset

def generate_sequence(model, start_seq, length, device):
    model.eval()
    generated = list(start_seq)
    input_seq = torch.tensor(start_seq).unsqueeze(0).to(device)
    hidden = None
    
    for _ in range(length):
        with torch.no_grad():
            logits, hidden = model(input_seq, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy().ravel()
            # Stochastische Auswahl mit Temperatur
            token = np.random.choice(len(probs), p=probs)
            generated.append(token)
            # Aktualisiere Input mit neuem Token
            input_seq = torch.tensor([generated[-model.seq_length:]]).to(device)
    
    return np.array(generated)

def get_unique_filename(base_name, extension):
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    torch.manual_seed(torch.seed())

    cfg = Config()
    dataset = MIDIDataset(cfg.midi_folder, cfg.seq_length)

    for _ in range(1000):
        model = MusicLSTM(cfg).to(cfg.device)
        ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state'])

        # Statt rein zufällig: zufälligen Ausschnitt aus Dataset als Seed wählen
        # Achtung: Dataset liefert (x,y) Paare, x ist seq_length lang
        idx = random.randint(0, len(dataset) - 1)
        start_seq, _ = dataset[idx]
        start_seq = start_seq.tolist()

        gen = generate_sequence(model, start_seq, 500, cfg.device)

        filename = get_unique_filename("generated_output", "mid")
        decode_sequence(gen, filename)
        print(f"{filename} erstellt")
