from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


from caddack.gnn.datasets import build_pyg_dataset
from caddack.gnn.geometry import ComplexExample
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


def _require_fusion_deps():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data  # type: ignore
        from torch_geometric.loader import DataLoader  # type: ignore
        from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Fusion training requires torch, torch-geometric, and scikit-learn."
        ) from exc
    return torch, F, Data, DataLoader, mean_absolute_error, r2_score


def _build_geo_pyg(ex: ComplexExample):
    """Build a torch-geometric Data for the geometry tower."""
    torch, _, Data, *_ = _require_fusion_deps()
    geo = ex.geo
    z = torch.tensor(geo.atomic_nums, dtype=torch.long)
    pos = torch.tensor(geo.positions, dtype=torch.float)
    return Data(z=z, pos=pos)


def train_fusion_from_complexes(
    complexes: List[ComplexExample],
    outdir: str | Path = "models/fusion",
    hidden_channels: int = 128,
    num_gine_layers: int = 3,
    num_geo_interactions: int = 3,
    num_rbf: int = 50,
    cutoff: float = 6.0,
    bayesian_hidden: Optional[List[int]] = None,
    prior_sigma: float = 1.0,
    kl_weight: float = 1.0,
    kl_warmup: int = 10,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
    split: str = "scaffold",
    mc_samples_eval: int = 30,
    no_aleatoric: bool = False,
) -> Dict[str, float]:
    """Train FusionAffinityNet on a list of ComplexExample objects.

    Uses scaffold split (via ligand SMILES) when available, else random split.
    Saves model.pt, metrics.json, and config.json to `outdir`.
    """
    torch, F, Data, DataLoader, mean_absolute_error, r2_score = _require_fusion_deps()
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet
    from caddack.gnn.bayes import elbo_loss

    if len(complexes) < 4:
        raise ValueError("Need at least 4 valid complexes to train.")

    # --- Build datasets ---
    valid = [ex for ex in complexes if ex.ligand_smiles is not None]
    use_scaffold = split == "scaffold" and len(valid) >= 4

    if use_scaffold:
        try:
            import pandas as pd
            from caddack.qsar.split import scaffold_split
            df = pd.DataFrame({"SMILES_canonical": [ex.ligand_smiles for ex in valid]})
            train_idx, test_idx = scaffold_split(df, smiles_col="SMILES_canonical",
                                                  test_size=test_size, seed=seed)
            complexes_to_use = valid
        except Exception:
            use_scaffold = False

    if not use_scaffold:
        complexes_to_use = complexes
        train_idx, test_idx = _split_indices(len(complexes_to_use), test_size=test_size, seed=seed)
        train_idx = list(train_idx)
        test_idx = list(test_idx)

    def _build_lig_pyg(ex: ComplexExample):
        try:
            g = smiles_to_graph_arrays(ex.ligand_smiles)
        except Exception:
            return None
        return to_pyg_data(g, y=ex.affinity)

    train_examples = [complexes_to_use[i] for i in train_idx]
    test_examples = [complexes_to_use[i] for i in test_idx]

    def make_pairs(examples):
        pairs = []
        for ex in examples:
            lig = _build_lig_pyg(ex)
            if lig is None:
                continue
            geo = _build_geo_pyg(ex)
            geo.y = torch.tensor([ex.affinity], dtype=torch.float)
            pairs.append((lig, geo))
        return pairs

    train_pairs = make_pairs(train_examples)
    test_pairs = make_pairs(test_examples)
    if len(train_pairs) < 2 or len(test_pairs) < 1:
        raise ValueError("Not enough valid pairs after building graphs.")

    n_train = len(train_pairs)

    def collate_pairs(batch):
        from torch_geometric.data import Batch  # type: ignore
        ligs, geos = zip(*batch)
        return Batch.from_data_list(list(ligs)), Batch.from_data_list(list(geos))

    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, collate_fn=collate_pairs)
    test_loader = DataLoader(test_pairs, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)

    # Infer feature dims from first example
    sample_lig = train_pairs[0][0]
    ligand_in_channels = sample_lig.x.shape[1]
    ligand_edge_dim = sample_lig.edge_attr.shape[1] if sample_lig.edge_attr.numel() else 7

    model = FusionAffinityNet.build(
        ligand_in_channels=ligand_in_channels,
        ligand_edge_dim=ligand_edge_dim,
        hidden_channels=hidden_channels,
        num_gine_layers=num_gine_layers,
        num_geo_interactions=num_geo_interactions,
        num_rbf=num_rbf,
        cutoff=cutoff,
        bayesian_hidden=bayesian_hidden or [256, 128],
        prior_sigma=prior_sigma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        beta = kl_weight * min(1.0, (epoch + 1) / max(kl_warmup, 1))
        for lig_batch, geo_batch in train_loader:
            optimizer.zero_grad()
            mu, log_var = model(lig_batch, geo_batch)
            y = geo_batch.y.view(-1)
            kl = model.kl()
            loss = elbo_loss(mu, log_var, y, kl, n_train=n_train,
                             kl_weight=beta, aleatoric=not no_aleatoric)
            loss.backward()
            optimizer.step()

    # Evaluation (MC sampling)
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for lig_batch, geo_batch in test_loader:
            pred_mean, _, _ = model.predict_with_uncertainty(
                lig_batch, geo_batch, n_samples=mc_samples_eval
            )
            all_preds.extend(pred_mean.cpu().numpy().tolist())
            all_truths.extend(geo_batch.y.view(-1).cpu().numpy().tolist())

    metrics = {
        "mae": float(mean_absolute_error(all_truths, all_preds)),
        "r2": float(r2_score(all_truths, all_preds)),
        "n_train": n_train,
        "n_test": len(test_pairs),
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    config = {
        "hidden_channels": hidden_channels,
        "num_gine_layers": num_gine_layers,
        "num_geo_interactions": num_geo_interactions,
        "num_rbf": num_rbf,
        "cutoff": cutoff,
        "bayesian_hidden": bayesian_hidden or [256, 128],
        "prior_sigma": prior_sigma,
        "ligand_in_channels": ligand_in_channels,
        "ligand_edge_dim": ligand_edge_dim,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return metrics
