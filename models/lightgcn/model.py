"""
LightGCN: Simplifying and Powering Graph Convolution Network (SIGIR 2020)
Used as the collaborative filtering recall channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_layers = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_layers.append(x)
        final = torch.stack(all_layers, dim=1).mean(dim=1)
        return final[:self.num_users], final[self.num_users:]

    def bpr_loss(self, users, pos_items, neg_items, edge_index, reg: float = 1e-4):
        user_embs, item_embs = self.forward(edge_index)
        u   = user_embs[users]
        pos = item_embs[pos_items]
        neg = item_embs[neg_items]

        pos_scores = (u * pos).sum(-1)
        neg_scores = (u * neg).sum(-1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = reg * (
            self.user_emb.weight[users].norm(2).pow(2) +
            self.item_emb.weight[pos_items].norm(2).pow(2) +
            self.item_emb.weight[neg_items].norm(2).pow(2)
        ) / len(users)

        return bpr + reg_loss

    @torch.no_grad()
    def recall_topk(self, edge_index, user_ids: torch.Tensor, k: int = 50):
        self.eval()
        user_embs, item_embs = self.forward(edge_index)
        scores = user_embs[user_ids] @ item_embs.T  # (B, num_items)
        return scores.topk(k, dim=-1).indices
