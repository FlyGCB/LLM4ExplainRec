"""
PyTorch Dataset classes for sequence-based recommendation.
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict


def build_user_sequences(df: pd.DataFrame, max_len: int = 50) -> dict:
    """Build {user_id: [item_id, ...]} sorted by timestamp."""
    df = df.sort_values(["user_id", "timestamp"])
    seqs = {}
    for uid, group in df.groupby("user_id"):
        seqs[uid] = group["item_id"].tolist()[-max_len:]
    return seqs


class SequenceDataset(Dataset):
    """
    For SASRec training.
    Each sample: (user, sequence, pos_item, neg_item)
    """
    def __init__(self, sequences: dict, num_items: int, max_len: int = 50):
        self.seqs = list(sequences.items())
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        uid, seq = self.seqs[idx]
        if len(seq) < 2:
            seq = seq + [0]  # pad
        # input: all but last; target: all but first
        input_seq = seq[:-1][-self.max_len:]
        target_seq = seq[1:][-self.max_len:]

        # Pad to max_len
        pad_len = self.max_len - len(input_seq)
        input_seq = [0] * pad_len + input_seq
        target_seq = [0] * pad_len + target_seq

        # Negative sample (random, excluding positives)
        pos_set = set(seq)
        neg = np.random.randint(1, self.num_items)
        while neg in pos_set:
            neg = np.random.randint(1, self.num_items)

        return {
            "user": torch.tensor(uid, dtype=torch.long),
            "input_seq": torch.tensor(input_seq, dtype=torch.long),
            "target_seq": torch.tensor(target_seq, dtype=torch.long),
            "neg_item": torch.tensor(neg, dtype=torch.long),
        }


class GCNDataset(Dataset):
    """
    For LightGCN training: (user, pos_item, neg_item) triples.
    """
    def __init__(self, df: pd.DataFrame, num_items: int):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.num_items = num_items
        self.user_items = defaultdict(set)
        for u, i in zip(self.users, self.items):
            self.user_items[u].add(i)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.items[idx]
        neg = np.random.randint(0, self.num_items)
        while neg in self.user_items[u]:
            neg = np.random.randint(0, self.num_items)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )
