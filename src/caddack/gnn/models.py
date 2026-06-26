from __future__ import annotations

import math
from typing import List


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


def _radius_graph_pure(pos, r, batch, loop=False):
    """Pure-PyTorch radius graph; fallback when pyg-lib is unavailable."""
    import torch
    rows, cols = [], []
    offset = 0
    for b in batch.unique():
        mask = batch == b
        sub = pos[mask]
        n = sub.shape[0]
        diff = sub.unsqueeze(0) - sub.unsqueeze(1)   # [n, n, 3]
        dist = diff.norm(dim=-1)                      # [n, n]
        adj = (dist <= r) if loop else ((dist <= r) & (dist > 0))
        src, dst = adj.nonzero(as_tuple=True)
        rows.append(src + offset)
        cols.append(dst + offset)
        offset += n
    if not rows:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def _require_torch_geo():
    """Subset of requirements for the geometry tower."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import global_add_pool  # type: ignore
        try:
            from torch_geometric.nn import radius_graph  # type: ignore
            # Probe whether radius_graph actually works (pyg>=2.4 requires pyg-lib)
            _test_pos = torch.zeros(2, 3)
            _test_bat = torch.zeros(2, dtype=torch.long)
            radius_graph(_test_pos, r=1.0, batch=_test_bat, loop=False)
        except (ImportError, Exception):
            radius_graph = _radius_graph_pure  # type: ignore
        try:
            from torch_scatter import scatter  # type: ignore
        except ImportError:
            from torch_geometric.utils import scatter  # type: ignore  # pyg>=2.3 built-in
    except Exception as exc:
        raise ImportError(
            "PyTorch + torch-geometric are required for the geometry tower."
        ) from exc
    return torch, nn, F, global_add_pool, radius_graph, scatter


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


class GaussianRBF:
    """Factory: Gaussian radial basis function expansion of interatomic distances."""

    @staticmethod
    def build(num_rbf: int = 50, cutoff: float = 6.0):
        torch, nn, F, *_ = _require_torch()

        class _GaussianRBF(nn.Module):
            def __init__(self):
                super().__init__()
                centers = torch.linspace(0.0, cutoff, num_rbf)
                self.register_buffer("centers", centers)
                self.width = (cutoff / num_rbf) ** 2

            def forward(self, dist):
                return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / self.width)

        return _GaussianRBF()


class GeometryTower:
    """SchNet-style continuous-filter conv tower over interatomic distances.

    SE(3)-invariant: operates on pairwise distances only, never raw coords.
    """

    @staticmethod
    def build(
        num_elements: int = 100,
        hidden_channels: int = 128,
        num_rbf: int = 50,
        num_interactions: int = 3,
        cutoff: float = 6.0,
    ):
        torch, nn, F, global_add_pool, radius_graph, scatter = _require_torch_geo()

        rbf_module = GaussianRBF.build(num_rbf=num_rbf, cutoff=cutoff)

        def _filter_net(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

        def _atom_mlp(dim):
            return nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        class _InteractionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.filter_net = _filter_net(num_rbf, hidden_channels)
                self.update_mlp = _atom_mlp(hidden_channels)
                self.res_proj = nn.Identity()

            def forward(self, h, edge_index, rbf, dist, cutoff_val):
                src, dst = edge_index
                # cosine envelope cutoff
                envelope = 0.5 * (torch.cos(math.pi * dist / cutoff_val) + 1.0)
                filt = self.filter_net(rbf) * envelope.unsqueeze(-1)  # [E, C]
                msg = h[src] * filt                                     # [E, C]
                agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")
                return h + self.update_mlp(agg)

        class _GeometryTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(num_elements + 1, hidden_channels)
                self.rbf = rbf_module
                self.blocks = nn.ModuleList([_InteractionBlock() for _ in range(num_interactions)])
                self.cutoff = cutoff

            def forward(self, z, pos, batch):
                h = self.embed(z.clamp(0, num_elements))
                edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
                src, dst = edge_index
                diff = pos[src] - pos[dst]
                dist = diff.norm(dim=-1)
                rbf = self.rbf(dist)
                for block in self.blocks:
                    h = block(h, edge_index, rbf, dist, self.cutoff)
                return global_add_pool(h, batch)

        return _GeometryTower()


class FusionAffinityNet:
    """Two-tower Bayesian fusion net for protein–ligand binding affinity."""

    @staticmethod
    def build(
        ligand_in_channels: int = 7,
        ligand_edge_dim: int = 7,
        hidden_channels: int = 128,
        num_gine_layers: int = 3,
        num_geo_interactions: int = 3,
        num_rbf: int = 50,
        cutoff: float = 6.0,
        num_elements: int = 100,
        bayesian_hidden: List[int] | None = None,
        prior_sigma: float = 1.0,
        dropout: float = 0.1,
    ):
        torch, nn, F, _, GINEConv, global_add_pool, _ = _require_torch()
        from caddack.gnn.bayes import BayesianMLP, elbo_loss  # noqa: F401

        if bayesian_hidden is None:
            bayesian_hidden = [256, 128]

        geo_tower = GeometryTower.build(
            num_elements=num_elements,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            num_interactions=num_geo_interactions,
            cutoff=cutoff,
        )

        def mlp(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        bayes_head = BayesianMLP.build(
            in_features=hidden_channels * 2,
            hidden_dims=bayesian_hidden,
            prior_sigma=prior_sigma,
            dropout=dropout,
        )

        class _FusionNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Tower 1: ligand 2D GINE
                self.lig_convs = nn.ModuleList()
                self.lig_convs.append(GINEConv(mlp(ligand_in_channels, hidden_channels), edge_dim=ligand_edge_dim))
                for _ in range(num_gine_layers - 1):
                    self.lig_convs.append(GINEConv(mlp(hidden_channels, hidden_channels), edge_dim=ligand_edge_dim))
                self.dropout_p = dropout

                # Tower 2: geometry
                self.geo_tower = geo_tower

                # Bayesian head
                self.bayes_head = bayes_head

            def _encode_ligand(self, lig_data):
                x, edge_index, edge_attr, batch = (
                    lig_data.x, lig_data.edge_index, lig_data.edge_attr, lig_data.batch
                )
                for conv in self.lig_convs:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                return global_add_pool(x, batch)

            def forward(self, lig_data, geo_data):
                h_lig = self._encode_ligand(lig_data)
                h_geo = self.geo_tower(geo_data.z, geo_data.pos, geo_data.batch)
                h = torch.cat([h_lig, h_geo], dim=-1)
                mu, log_var = self.bayes_head(h)
                return mu, log_var

            def kl(self):
                return self.bayes_head.kl()

            def predict_with_uncertainty(self, lig_data, geo_data, n_samples: int = 30):
                """Return (mean, epistemic_std, aleatoric_std) via MC sampling."""
                self.eval()
                mus, vars_ = [], []
                with torch.no_grad():
                    for _ in range(n_samples):
                        mu, log_var = self.forward(lig_data, geo_data)
                        mus.append(mu)
                        vars_.append(torch.exp(log_var))
                mus_t = torch.stack(mus, dim=0)          # [T, B]
                vars_t = torch.stack(vars_, dim=0)        # [T, B]
                pred_mean = mus_t.mean(0)
                epistemic = mus_t.var(0).sqrt()
                aleatoric = vars_t.mean(0).sqrt()
                return pred_mean, epistemic, aleatoric

        return _FusionNet()


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
