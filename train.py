import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from dataset import MIDIDataset
from model import MusicLSTM
import numpy as np
import time

if __name__ == "__main__":
    cfg = Config()
    dataset = MIDIDataset(cfg.midi_folder, cfg.seq_length)
    
    # Debug: Ausgabe der Sequenzlängen
    seq_lengths = [len(seq) for seq in dataset.sequences]
    print(f"Dataset loaded: {len(dataset.sequences)} sequences")
    print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Avg: {np.mean(seq_lengths):.1f}")
    
    num_workers = 2 if cfg.device == "cpu" else 0
    loader = DataLoader(dataset, 
                       batch_size=cfg.batch_size, 
                       shuffle=True, 
                       num_workers=num_workers,
                       pin_memory=True if cfg.device == "cuda" else False)

    model = MusicLSTM(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=129)  # Ignoriere CHORD_TOKEN im Loss
    
    # Gradient Clipping
    clip_value = 1.0

    use_mixed_precision = (cfg.device == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None #TODO: ADD torch.amp.GradScaler('cuda', args...)

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()

            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, _ = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

            total_loss += loss.item()
            
            # Progress-Anzeige
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        # Checkpoint-Logik
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss,
                'config': cfg.__dict__
            }, cfg.checkpoint_path)
            print(f"🔥 Saved BEST model (loss: {best_loss:.4f})")
        else:
            # Speichere alle 5 Epochen
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': avg_loss
                }, f"checkpoint_epoch{epoch+1}.pth")
                print(f"💾 Saved periodic checkpoint")