"""
SASRec: Self-Attentive Sequential Recommendation (ICDM 2018)
Used as one of the multi-channel recall models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SASRec(nn.Module):
    def __init__(self, num_items: int, hidden_size: int = 64,
                 num_heads: int = 2, num_layers: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, hidden_size)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_seq):
        """
        input_seq: (B, L) item ids
        returns:   (B, L, D) sequence representations
        """
        B, L = input_seq.shape
        positions = torch.arange(1, L + 1, device=input_seq.device).unsqueeze(0)
        x = self.emb_drop(self.item_emb(input_seq) + self.pos_emb(positions))

        # Causal mask: only attend to past
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # (B, L, D)

    def predict(self, input_seq, item_ids):
        """
        Score a set of candidate items against the last position.
        input_seq: (B, L)
        item_ids:  (B, K) or (K,)
        returns:   (B, K) scores
        """
        seq_repr = self.forward(input_seq)[:, -1, :]  # (B, D)
        item_embs = self.item_emb(item_ids)           # (B, K, D) or (K, D)
        if item_embs.dim() == 2:
            return (seq_repr @ item_embs.T)
        return (seq_repr.unsqueeze(1) * item_embs).sum(-1)

    def bpr_loss(self, input_seq, pos_items, neg_items):
        seq_repr = self.forward(input_seq)  # (B, L, D)
        # Only use positions with actual targets (non-zero)
        # Flatten and compute BPR over all valid positions
        pos_emb = self.item_emb(pos_items)  # (B, L, D)
        neg_emb = self.item_emb(neg_items.unsqueeze(1).expand_as(pos_items))

        pos_scores = (seq_repr * pos_emb).sum(-1)  # (B, L)
        neg_scores = (seq_repr * neg_emb).sum(-1)

        mask = (pos_items != 0).float()
        loss = -F.logsigmoid(pos_scores - neg_scores) * mask
        return loss.sum() / mask.sum()


class SASRecBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d)
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask.float() * -1e9)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x
