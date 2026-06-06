"""
LightGCN training script — optimized for large datasets (3M+ interactions).
Usage: python -m models.lightgcn.train --config configs/lightgcn.yaml
"""
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from models.lightgcn.model import LightGCN


# ── Dataset ──────────────────────────────────────────────────────────────────

class GCNDataset(Dataset):
    """
    BPR training triples (user, pos_item, neg_item).
    Negative sampling is done lazily per batch to avoid huge memory overhead.
    """
    def __init__(self, df: pd.DataFrame, num_items: int):
        self.users    = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items    = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.num_items = num_items

        # Build positive set per user for valid negative sampling
        print("  Building user->item positive sets...")
        user_items = {}
        for u, i in zip(df["user_id"].values, df["item_id"].values):
            if u not in user_items:
                user_items[u] = set()
            user_items[u].add(i)
        self.user_items = user_items
        print(f"  Done. {len(user_items)} users indexed.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u   = self.users[idx].item()
        pos = self.items[idx].item()
        # Fast negative sampling: retry max 10 times, then give up
        neg = np.random.randint(0, self.num_items)
        for _ in range(10):
            if neg not in self.user_items[u]:
                break
            neg = np.random.randint(0, self.num_items)
        return (
            torch.tensor(u,   dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ── Edge index ────────────────────────────────────────────────────────────────

def build_edge_index(df: pd.DataFrame, num_users: int, device: str) -> torch.Tensor:
    """Bidirectional edge_index for user-item bipartite graph."""
    users = torch.tensor(df["user_id"].values, dtype=torch.long)
    items = torch.tensor(df["item_id"].values, dtype=torch.long) + num_users
    edge_u2i = torch.stack([users, items], dim=0)
    edge_i2u = torch.stack([items, users], dim=0)
    return torch.cat([edge_u2i, edge_i2u], dim=1).to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_recall(model, edge_index, val_data, num_items, k, device, batch_size=512):
    """Batched evaluation to avoid OOM on large user sets."""
    model.eval()
    hits = 0
    with torch.no_grad():
        user_embs, item_embs = model(edge_index)
        for start in range(0, len(val_data), batch_size):
            batch    = val_data[start:start + batch_size]
            uids     = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
            scores   = user_embs[uids] @ item_embs.T          # (B, num_items)
            topk_idx = scores.topk(k, dim=-1).indices         # (B, k)
            for i, (_, true_item) in enumerate(batch):
                if true_item in topk_idx[i].tolist():
                    hits += 1
    return hits / len(val_data)


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg_path: str = "configs/lightgcn.yaml"):
    cfg    = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    df       = pd.read_parquet("data/processed/interactions.parquet")
    val_df   = pd.read_parquet("data/processed/val.parquet")
    meta     = json.load(open("data/processed/meta.json"))
    num_users, num_items = meta["num_users"], meta["num_items"]
    print(f"  {len(df):,} interactions | {num_users:,} users | {num_items:,} items")

    print("Building edge index...")
    edge_index = build_edge_index(df, num_users, device)
    print(f"  edge_index shape: {edge_index.shape}")

    print("Building dataset...")
    dataset  = GCNDataset(df, num_items)
    loader   = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,      # safer on JupyterHub clusters
        pin_memory=(device == "cuda"),
    )
    print(f"  {len(loader)} batches per epoch")

    model = LightGCN(
        num_users, num_items,
        cfg.model.embedding_dim,
        cfg.model.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    val_data  = list(val_df.itertuples(index=False, name=None))

    best_recall = 0
    Path("checkpoints").mkdir(exist_ok=True)

    print(f"\nStarting training for {cfg.train.epochs} epochs...\n")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:4d}/{cfg.train.epochs}", leave=False)
        for u, pos, neg in pbar:
            optimizer.zero_grad()
            loss = model.bpr_loss(
                u.to(device), pos.to(device), neg.to(device),
                edge_index, cfg.model.reg_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        if epoch % 100 == 0:
            recall = eval_recall(
                model, edge_index, val_data, num_items,
                cfg.train.recall_topk, device,
            )
            print(f"Epoch {epoch:4d} | Loss {avg_loss:.4f} | Recall@{cfg.train.recall_topk}: {recall:.4f}")
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), "checkpoints/lightgcn_best.pt")
                print(f"  ✓ New best saved.")

    print(f"\nBest Recall@{cfg.train.recall_topk}: {best_recall:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/lightgcn.yaml")
    args = p.parse_args()
    train(args.config)
