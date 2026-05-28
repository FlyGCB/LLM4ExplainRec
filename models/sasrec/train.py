"""
SASRec training script.
Usage: python -m models.sasrec.train --config configs/sasrec.yaml
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path
import json

from models.sasrec.model import SASRec
from data.dataset import SequenceDataset, build_user_sequences
import pandas as pd


def recall_at_k(model, sequences, val_data, num_items, k=50, device="cuda"):
    model.eval()
    hits = 0
    with torch.no_grad():
        for uid, true_item in val_data:
            seq = sequences.get(uid, [])
            if not seq:
                continue
            input_seq = torch.tensor(seq[-50:], dtype=torch.long).unsqueeze(0).to(device)
            # pad
            pad = torch.zeros(1, 50 - input_seq.shape[1], dtype=torch.long).to(device)
            input_seq = torch.cat([pad, input_seq], dim=1)

            all_items = torch.arange(1, num_items + 1, device=device)
            scores = model.predict(input_seq, all_items).squeeze(0)
            topk = scores.topk(k).indices + 1
            if true_item in topk.tolist():
                hits += 1
    return hits / len(val_data)


def train(cfg_path: str = "configs/sasrec.yaml"):
    cfg = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet("data/processed/interactions.parquet")
    meta = json.load(open("data/processed/meta.json"))
    val_df = pd.read_parquet("data/processed/val.parquet")

    num_users = meta["num_users"]
    num_items = meta["num_items"]

    sequences = build_user_sequences(df, max_len=cfg.model.max_seq_len)
    dataset   = SequenceDataset(sequences, num_items, cfg.model.max_seq_len)
    loader    = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)

    model = SASRec(
        num_items=num_items,
        hidden_size=cfg.model.hidden_size,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    val_data  = list(val_df.itertuples(index=False, name=None))

    best_recall = 0
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model.bpr_loss(
                batch["input_seq"].to(device),
                batch["target_seq"].to(device),
                batch["neg_item"].to(device),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            recall = recall_at_k(model, sequences, val_data, num_items, k=cfg.train.recall_topk, device=device)
            print(f"Epoch {epoch:3d} | Loss {total_loss/len(loader):.4f} | Recall@{cfg.train.recall_topk}: {recall:.4f}")
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), "checkpoints/sasrec_best.pt")

    print(f"Best Recall@{cfg.train.recall_topk}: {best_recall:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sasrec.yaml")
    args = p.parse_args()
    train(args.config)
