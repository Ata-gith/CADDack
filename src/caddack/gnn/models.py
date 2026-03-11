from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import (  # type: ignore
            GCNConv,
            GINEConv,
            global_add_pool,
            global_mean_pool,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch + torch-geometric are required for GNN models."
        ) from exc
    return torch, nn, F, GCNConv, GINEConv, global_add_pool, global_mean_pool


class MolecularGCN:
    """Factory wrapper exposing a torch.nn.Module when torch is available."""

    @staticmethod
    def build(
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        task: str = "classification",
    ):
        torch, nn, F, GCNConv, _, _, global_mean_pool = _require_torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.task = task
                self.dropout = dropout
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(in_channels, hidden_channels))
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.head = nn.Linear(hidden_channels, 1)

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = global_mean_pool(x, batch)
                logits = self.head(x).view(-1)
                return logits

        return _Model()


class MolecularGINE:
    @staticmethod
    def build(
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        task: str = "classification",
    ):
        torch, nn, F, _, GINEConv, global_add_pool, _ = _require_torch()

        def mlp(in_dim: int, out_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.task = task
                self.dropout = dropout
                self.convs = nn.ModuleList()
                self.convs.append(GINEConv(mlp(in_channels, hidden_channels), edge_dim=edge_dim))
                for _ in range(num_layers - 1):
                    self.convs.append(GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=edge_dim))
                self.head = nn.Linear(hidden_channels, 1)

            def forward(self, data):
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                for conv in self.convs:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = global_add_pool(x, batch)
                logits = self.head(x).view(-1)
                return logits

        return _Model()
