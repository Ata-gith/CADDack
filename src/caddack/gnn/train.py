from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence


from caddack.gnn.datasets import build_pyg_dataset
from caddack.gnn.models import MolecularGCN, MolecularGINE


def _require_torch_geometric():
    try:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
        from torch_geometric.loader import DataLoader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Training requires torch, torch-geometric, and scikit-learn."
        ) from exc
    return torch, F, DataLoader, accuracy_score, roc_auc_score, mean_absolute_error, r2_score


def _split_indices(n: int, test_size: float = 0.2, seed: int = 42):
    torch, *_ = _require_torch_geometric()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cut = max(1, int(round(n * (1 - test_size))))
    return order[:cut], order[cut:]


def _prepare_frame(df, smiles_col: str, target_col: str, task: str, positive_threshold: float | None):
    if smiles_col not in df or target_col not in df:
        raise ValueError(f"Missing required columns {smiles_col!r} and/or {target_col!r}")

    data_df = df[[smiles_col, target_col]].dropna().copy()
    if task == "classification" and positive_threshold is not None:
        data_df[target_col] = (data_df[target_col].astype(float) >= float(positive_threshold)).astype(int)
    return data_df


def train_from_csv(
    csv_path: str | Path,
    smiles_col: str,
    target_col: str,
    outdir: str | Path = "models/gnn",
    model_name: str = "gine",
    task: str = "classification",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_channels: int = 128,
    num_layers: int = 3,
    test_size: float = 0.2,
    seed: int = 42,
    positive_threshold: float | None = None,
) -> Dict[str, float]:
    torch, F, DataLoader, accuracy_score, roc_auc_score, mean_absolute_error, r2_score = _require_torch_geometric()
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required to read training CSV files.") from exc

    df = pd.read_csv(csv_path)
    data_df = _prepare_frame(df, smiles_col=smiles_col, target_col=target_col, task=task, positive_threshold=positive_threshold)
    dataset = build_pyg_dataset(data_df[smiles_col].tolist(), data_df[target_col].tolist())
    if len(dataset) < 4:
        raise ValueError("Need at least 4 valid molecules to train/evaluate")

    train_idx, test_idx = _split_indices(len(dataset), test_size=test_size, seed=seed)
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    in_channels = dataset[0].x.shape[1]
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr.numel() else 7
    if model_name.lower() == "gcn":
        model = MolecularGCN.build(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            task=task,
        )
    else:
        model = MolecularGINE.build(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            task=task,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(batch):
        logits = model(batch)
        y = batch.y.view(-1)
        if task == "classification":
            return F.binary_cross_entropy_with_logits(logits, y)
        return F.mse_loss(logits, y)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = step(batch)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)
            y = batch.y.view(-1)
            if task == "classification":
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy().tolist())
            else:
                preds.extend(logits.cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())

    if task == "classification":
        labels = [1 if p >= 0.5 else 0 for p in preds]
        metrics = {
            "auc_roc": float(roc_auc_score(truths, preds)),
            "accuracy": float(accuracy_score(truths, labels)),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(truths, preds)),
            "r2": float(r2_score(truths, preds)),
        }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
