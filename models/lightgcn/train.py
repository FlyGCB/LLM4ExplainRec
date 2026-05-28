"""
LightGCN training script.
Usage: python -m models.lightgcn.train --config configs/lightgcn.yaml
"""
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path

from models.lightgcn.model import LightGCN
from data.dataset import GCNDataset


def build_edge_index(df: pd.DataFrame, num_users: int, device: str) -> torch.Tensor:
    """
    Build bidirectional edge_index for the user-item bipartite graph.
    Item ids are offset by num_users in the joint node space.
    """
    users = torch.tensor(df["user_id"].values, dtype=torch.long)
    items = torch.tensor(df["item_id"].values, dtype=torch.long) + num_users
    edge_u2i = torch.stack([users, items], dim=0)
    edge_i2u = torch.stack([items, users], dim=0)
    return torch.cat([edge_u2i, edge_i2u], dim=1).to(device)


def eval_recall(model, edge_index, val_data, num_items, k, device):
    model.eval()
    hits = 0
    with torch.no_grad():
        user_ids = torch.tensor([u for u, _ in val_data], dtype=torch.long, device=device)
        topk = model.recall_topk(edge_index, user_ids, k=k)  # (N, k)
        for i, (_, true_item) in enumerate(val_data):
            if true_item in topk[i].tolist():
                hits += 1
    return hits / len(val_data)


def train(cfg_path: str = "configs/lightgcn.yaml"):
    cfg    = OmegaConf.load(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df      = pd.read_parquet("data/processed/interactions.parquet")
    val_df  = pd.read_parquet("data/processed/val.parquet")
    meta    = json.load(open("data/processed/meta.json"))
    num_users, num_items = meta["num_users"], meta["num_items"]

    edge_index = build_edge_index(df, num_users, device)
    dataset    = GCNDataset(df, num_items)
    loader     = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)

    model = LightGCN(num_users, num_items,
                     cfg.model.embedding_dim, cfg.model.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    val_data  = list(val_df.itertuples(index=False, name=None))

    best_recall = 0
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0
        for u, pos, neg in loader:
            optimizer.zero_grad()
            loss = model.bpr_loss(u.to(device), pos.to(device), neg.to(device),
                                  edge_index, cfg.model.reg_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 100 == 0:
            recall = eval_recall(model, edge_index, val_data, num_items,
                                 cfg.train.recall_topk, device)
            print(f"Epoch {epoch:4d} | Loss {total_loss/len(loader):.4f} | Recall@{cfg.train.recall_topk}: {recall:.4f}")
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), "checkpoints/lightgcn_best.pt")

    print(f"Best Recall@{cfg.train.recall_topk}: {best_recall:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/lightgcn.yaml")
    args = p.parse_args()
    train(args.config)
