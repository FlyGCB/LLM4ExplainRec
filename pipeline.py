"""
Full recommendation pipeline: multi-channel recall → LLM rerank & explain.
"""
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from models.sasrec.model import SASRec
from models.lightgcn.model import LightGCN
from data.dataset import build_user_sequences


@dataclass
class RecallResult:
    history: list[str]       # item titles
    candidates: list[dict]   # [{item_id, title, category, price}]


class RecommendationPipeline:
    def __init__(self, sasrec: SASRec, lightgcn: LightGCN,
                 sequences: dict, item_meta: pd.DataFrame,
                 edge_index: torch.Tensor, device: str = "cpu"):
        self.sasrec    = sasrec
        self.lightgcn  = lightgcn
        self.sequences = sequences
        self.item_meta = item_meta.set_index("item_id")
        self.edge_index = edge_index
        self.device    = device

    def recall(self, user_id: int, topk: int = 50) -> RecallResult:
        seq = self.sequences.get(user_id, [])
        if not seq:
            return RecallResult([], [])

        # SASRec recall
        input_seq = torch.tensor(seq[-50:], dtype=torch.long).unsqueeze(0).to(self.device)
        pad = torch.zeros(1, 50 - input_seq.shape[1], dtype=torch.long).to(self.device)
        input_seq = torch.cat([pad, input_seq], dim=1)
        all_items = torch.arange(1, len(self.item_meta) + 1, device=self.device)
        with torch.no_grad():
            sasrec_scores = self.sasrec.predict(input_seq, all_items).squeeze(0)
        sasrec_topk = sasrec_scores.topk(topk).indices.tolist()

        # LightGCN recall
        uid_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            gcn_topk = self.lightgcn.recall_topk(self.edge_index, uid_tensor, k=topk).squeeze(0).tolist()

        # Merge (union, deduplicated)
        merged = list(dict.fromkeys(sasrec_topk + gcn_topk))[:topk]

        # Build candidate dicts
        history_titles = [self._item_title(i) for i in seq[-10:]]
        candidates = [self._item_dict(i) for i in merged]

        return RecallResult(history=history_titles, candidates=candidates)

    def _item_title(self, item_id: int) -> str:
        try:
            return self.item_meta.loc[item_id, "title"]
        except KeyError:
            return f"item_{item_id}"

    def _item_dict(self, item_id: int) -> dict:
        try:
            row = self.item_meta.loc[item_id]
            return {"item_id": str(item_id), "title": row.get("title", ""),
                    "category": row.get("category", ""), "price": row.get("price", None)}
        except KeyError:
            return {"item_id": str(item_id), "title": f"item_{item_id}", "category": "", "price": None}


def build_pipeline(device: str = "cpu") -> RecommendationPipeline:
    meta    = json.load(open("data/processed/meta.json"))
    df      = pd.read_parquet("data/processed/interactions.parquet")
    sequences = build_user_sequences(df)

    # Load item metadata (titles, categories) if available
    meta_path = Path("data/processed/item_meta.parquet")
    item_meta = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame(
        {"item_id": range(meta["num_items"]), "title": [f"item_{i}" for i in range(meta["num_items"])],
         "category": ["unknown"] * meta["num_items"]})

    # Load SASRec
    sasrec = SASRec(num_items=meta["num_items"]).to(device)
    if Path("checkpoints/sasrec_best.pt").exists():
        sasrec.load_state_dict(torch.load("checkpoints/sasrec_best.pt", map_location=device))
    sasrec.eval()

    # Load LightGCN
    lightgcn = LightGCN(meta["num_users"], meta["num_items"]).to(device)
    if Path("checkpoints/lightgcn_best.pt").exists():
        lightgcn.load_state_dict(torch.load("checkpoints/lightgcn_best.pt", map_location=device))
    lightgcn.eval()

    # Build edge index
    from models.lightgcn.train import build_edge_index
    edge_index = build_edge_index(df, meta["num_users"], device)

    return RecommendationPipeline(sasrec, lightgcn, sequences, item_meta, edge_index, device)
