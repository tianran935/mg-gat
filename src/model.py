from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MGGAT(nn.Module):
    def __init__(self, num_users: int, num_businesses: int, user_feat_dim: int, business_feat_dim: int, hidden_dim: int = 64, latent_dim: int = 64, num_business_graphs: int = 3, interpretable: bool = True) -> None:
        super().__init__()
        self.interpretable = interpretable
        if interpretable:
            self.user_linear = nn.Linear(user_feat_dim, hidden_dim, bias=False)
            self.business_linear = nn.Linear(business_feat_dim, hidden_dim, bias=False)
        else:
            self.user_linear = nn.Sequential(nn.Linear(user_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.business_linear = nn.Sequential(nn.Linear(business_feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.user_attention = nn.Parameter(torch.randn(2 * hidden_dim) * 0.02)
        self.business_attention = nn.Parameter(torch.randn(2 * hidden_dim) * 0.02)
        self.graph_weight = nn.Parameter(torch.zeros(num_business_graphs))
        self.user_dense_graph = nn.Linear(hidden_dim, hidden_dim)
        self.user_dense_skip = nn.Linear(user_feat_dim, hidden_dim)
        self.business_dense_graph = nn.Linear(hidden_dim, hidden_dim)
        self.business_dense_skip = nn.Linear(business_feat_dim, hidden_dim)
        self.user_out = nn.Linear(hidden_dim, latent_dim)
        self.business_out = nn.Linear(hidden_dim, latent_dim)
        self.user_base = nn.Embedding(num_users, latent_dim)
        self.business_base = nn.Embedding(num_businesses, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.business_bias = nn.Embedding(num_businesses, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_features: torch.Tensor, business_features: torch.Tensor, user_edges: torch.Tensor, business_edges: torch.Tensor, business_edge_type: torch.Tensor, user_idx: torch.Tensor, business_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_emb, business_emb = self.compute_embeddings(user_features, business_features, user_edges, business_edges, business_edge_type)
        score = (user_emb[user_idx] * business_emb[business_idx]).sum(dim=1)
        score = score + self.user_bias(user_idx).squeeze(-1) + self.business_bias(business_idx).squeeze(-1) + self.global_bias
        pred = 4.0 * torch.sigmoid(score) + 1.0
        return pred, user_emb, business_emb

    def compute_embeddings(self, user_features, business_features, user_edges, business_edges, business_edge_type):
        u1 = self.user_linear(user_features)
        b1 = self.business_linear(business_features)
        u2 = self._aggregate(u1, user_edges, self.user_attention, None)
        b2 = self._aggregate(b1, business_edges, self.business_attention, business_edge_type)
        u3 = torch.relu(self.user_dense_graph(u2) + self.user_dense_skip(user_features))
        b3 = torch.relu(self.business_dense_graph(b2) + self.business_dense_skip(business_features))
        user_emb = torch.relu(self.user_out(u3)) + self.user_base.weight
        business_emb = torch.relu(self.business_out(b3)) + self.business_base.weight
        return user_emb, business_emb

    def _aggregate(self, features, edges, attention_vector, edge_type: Optional[torch.Tensor]):
        if edges.numel() == 0:
            return torch.zeros_like(features)
        src = edges[:, 0]
        dst = edges[:, 1]
        concat = torch.cat([features[src], features[dst]], dim=1)
        scores = self.leaky_relu((concat * attention_vector.unsqueeze(0)).sum(dim=1))
        if edge_type is not None and edge_type.numel() > 0:
            scores = scores + self.graph_weight[edge_type]
        weights = torch.sigmoid(scores)
        denom = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
        denom.index_add_(0, src, weights)
        weights = weights / denom[src].clamp_min(1e-8)
        out = torch.zeros_like(features)
        out.index_add_(0, src, weights.unsqueeze(1) * features[dst])
        return out

    def graph_regularization(self, user_edges, business_edges, theta2: float):
        reg = 0.0
        if user_edges.numel() > 0:
            reg = reg + (self.user_base.weight[user_edges[:, 0]] - self.user_base.weight[user_edges[:, 1]]).pow(2).sum(dim=1).mean()
        if business_edges.numel() > 0:
            reg = reg + (self.business_base.weight[business_edges[:, 0]] - self.business_base.weight[business_edges[:, 1]]).pow(2).sum(dim=1).mean()
        reg = reg + theta2 * (self.user_base.weight.pow(2).mean() + self.business_base.weight.pow(2).mean())
        return reg
