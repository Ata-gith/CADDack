#!/usr/bin/env python3
"""Benchmark script for FusionAffinityNet.

Covers three areas:
  1. Speed / throughput  — forward pass, MC sampling latency at varying n_samples
  2. Calibration         — empirical coverage vs nominal Gaussian intervals (ECE proxy)
  3. Accuracy vs baseline— FusionAffinityNet MAE/R² vs mean-predictor and GINE-only

Run with:
    python scripts/benchmark_fusion.py [--n-complexes 64] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_complexes(n: int, seed: int = 42):
    """Generate synthetic ComplexExample objects with varied geometry and affinity."""
    from caddack.gnn.geometry import ComplexExample, GeometryRecord

    rng = random.Random(seed)
    smiles_pool = [
        "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCO", "c1ccncc1",
        "CC(C)O", "c1ccc(N)cc1", "CCC(=O)O", "c1ccoc1",
        "c1ccc(F)cc1", "CC(=O)N", "CCCN", "c1ccsc1", "CCCl",
    ]
    complexes = []
    for i in range(n):
        n_atoms = rng.randint(5, 12)
        positions = [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))
                     for _ in range(n_atoms)]
        atomic_nums = [rng.choice([6, 7, 8, 6, 6]) for _ in range(n_atoms)]
        n_lig = max(1, n_atoms // 3)
        n_poc = n_atoms - n_lig

        geo = GeometryRecord(
            atomic_nums=atomic_nums,
            positions=positions,
            n_ligand=n_lig,
            n_pocket=n_poc,
        )
        # affinity with a simple linear trend over i + noise
        affinity = 4.0 + (i / n) * 4.0 + rng.gauss(0, 0.3)
        complexes.append(ComplexExample(
            pdb_id=f"SYN{i:04d}",
            affinity=affinity,
            ligand_smiles=smiles_pool[i % len(smiles_pool)],
            geo=geo,
        ))
    return complexes


# ---------------------------------------------------------------------------
# Section 1: Speed benchmarks
# ---------------------------------------------------------------------------

def bench_speed(model, lig_batch, geo_batch, device, n_samples_list=(5, 10, 30, 50)):
    import torch

    model.eval()
    model.to(device)
    lig_batch = lig_batch.to(device)
    geo_batch = geo_batch.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            model(lig_batch, geo_batch)

    # --- Single forward pass ---
    N = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(lig_batch, geo_batch)
    fwd_ms = (time.perf_counter() - t0) / N * 1000

    # --- MC sampling at varying n_samples ---
    mc_times = {}
    for ns in n_samples_list:
        t0 = time.perf_counter()
        with torch.no_grad():
            model.predict_with_uncertainty(lig_batch, geo_batch, n_samples=ns)
        mc_times[ns] = (time.perf_counter() - t0) * 1000

    return fwd_ms, mc_times


# ---------------------------------------------------------------------------
# Section 2: Calibration (empirical coverage)
# ---------------------------------------------------------------------------

def calibration_ece(
    preds: List[float],
    epistemic_stds: List[float],
    truths: List[float],
    n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """Gaussian empirical coverage ECE proxy.

    For each nominal confidence level p, check fraction of true values
    within ±z_p * sigma. ECE = mean |observed - expected| over levels.
    """
    import scipy.stats as stats  # type: ignore

    levels = [i / n_bins for i in range(1, n_bins + 1)]
    bins = []
    for p in levels:
        z = stats.norm.ppf((1 + p) / 2)
        covered = sum(
            abs(t - m) <= z * s
            for t, m, s in zip(truths, preds, epistemic_stds)
        )
        observed = covered / len(truths)
        bins.append({"nominal": p, "observed": observed, "error": abs(observed - p)})

    ece = sum(b["error"] for b in bins) / n_bins
    return ece, bins


# ---------------------------------------------------------------------------
# Section 3: Accuracy vs baseline
# ---------------------------------------------------------------------------

def train_baseline(
    train_smiles, train_affinities, test_smiles, test_affinities
) -> Tuple[float, float]:
    """GINE-only baseline (no geometry tower, no Bayes) trained for 20 epochs."""
    import torch
    from torch_geometric.loader import DataLoader
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import MolecularGINE
    import torch.nn.functional as F

    train_set = [to_pyg_data(smiles_to_graph_arrays(s), y=y)
                 for s, y in zip(train_smiles, train_affinities)]
    test_set = [to_pyg_data(smiles_to_graph_arrays(s), y=y)
                for s, y in zip(test_smiles, test_affinities)]

    model = MolecularGINE.build(in_channels=7, edge_dim=7, hidden_channels=32,
                                 num_layers=2, task="regression")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_set, batch_size=8, shuffle=True)

    model.train()
    for _ in range(20):
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()
            opt.step()

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for d in test_set:
            from torch_geometric.data import Batch
            b = Batch.from_data_list([d])
            preds.append(model(b).item())
            truths.append(d.y.item())

    mae = sum(abs(p - t) for p, t in zip(preds, truths)) / len(truths)
    mean_t = sum(truths) / len(truths)
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, truths))
    ss_tot = sum((t - mean_t) ** 2 for t in truths)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark FusionAffinityNet")
    parser.add_argument("--n-complexes", type=int, default=64,
                        help="Synthetic complexes to generate (default 64)")
    parser.add_argument("--device", default="cpu", help="torch device (default cpu)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs for fusion training (default 20)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out", default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    try:
        import torch
        from torch_geometric.data import Data, Batch
        from caddack.gnn.models import FusionAffinityNet
        from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
        from caddack.gnn.train import train_fusion_from_complexes
    except ImportError as e:
        print(f"ERROR: missing dependencies — {e}")
        print("Install with: pip install -e '.[gnn]'")
        return

    device = torch.device(args.device)
    n = args.n_complexes
    print(f"\n=== CADDack FusionAffinityNet Benchmark ===")
    print(f"  Complexes : {n}")
    print(f"  Device    : {device}")
    print(f"  Hidden    : {args.hidden}")
    print()

    # ---- build synthetic data ----
    complexes = _make_complexes(n, seed=0)

    # ---- split 80/20 ----
    split_at = int(n * 0.8)
    train_cx = complexes[:split_at]
    test_cx = complexes[split_at:]

    # ----------------------------------------------------------------
    # SECTION 1 — Speed
    # ----------------------------------------------------------------
    print("--- [1/3] Speed benchmarks ---")
    mini_model = FusionAffinityNet.build(
        ligand_in_channels=7, ligand_edge_dim=7,
        hidden_channels=args.hidden,
        num_gine_layers=2, num_geo_interactions=2,
        num_rbf=16, cutoff=5.0, bayesian_hidden=[args.hidden],
    ).to(device)

    # Batch of 8 synthetic graphs
    g = smiles_to_graph_arrays("CCO")
    lig_data = to_pyg_data(g, y=5.0)
    lig_batch = Batch.from_data_list([lig_data] * 8).to(device)

    z = torch.tensor([6, 7, 8, 6, 6, 6], dtype=torch.long)
    pos = torch.tensor([[0, 0, 0], [1.5, 0, 0], [3, 0, 0],
                         [0, 1.5, 0], [1.5, 1.5, 0], [3, 1.5, 0]], dtype=torch.float)
    geo_data = Data(z=z, pos=pos)
    geo_batch = Batch.from_data_list([geo_data] * 8).to(device)

    fwd_ms, mc_times = bench_speed(mini_model, lig_batch, geo_batch, device)
    print(f"  Forward pass (batch=8)   : {fwd_ms:.2f} ms")
    for ns, t in mc_times.items():
        print(f"  MC sampling (n={ns:3d})     : {t:.2f} ms")
    print()

    # ----------------------------------------------------------------
    # SECTION 2 — Train and calibrate
    # ----------------------------------------------------------------
    print("--- [2/3] Training fusion model for calibration + accuracy ---")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        metrics = train_fusion_from_complexes(
            complexes=train_cx,
            outdir=tmpdir,
            hidden_channels=args.hidden,
            num_gine_layers=2,
            num_geo_interactions=2,
            num_rbf=16,
            cutoff=5.0,
            bayesian_hidden=[args.hidden],
            epochs=args.epochs,
            batch_size=8,
            split="random",
            mc_samples_eval=10,
            kl_warmup=5,
        )
        train_sec = time.perf_counter() - t0

        # load trained model for calibration eval on test set
        import torch
        from caddack.gnn.models import FusionAffinityNet
        from caddack.gnn.train import _build_geo_pyg
        from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
        from torch_geometric.data import Batch

        config_path = os.path.join(tmpdir, "config.json")
        config = json.loads(open(config_path).read())

        trained_model = FusionAffinityNet.build(
            ligand_in_channels=config["ligand_in_channels"],
            ligand_edge_dim=config["ligand_edge_dim"],
            hidden_channels=config["hidden_channels"],
            num_gine_layers=config["num_gine_layers"],
            num_geo_interactions=config["num_geo_interactions"],
            num_rbf=config["num_rbf"],
            cutoff=config["cutoff"],
            bayesian_hidden=config["bayesian_hidden"],
        )
        trained_model.load_state_dict(torch.load(os.path.join(tmpdir, "model.pt"),
                                                   map_location="cpu"))
        trained_model.eval()

    preds, eps_stds, ale_stds, truths = [], [], [], []
    for ex in test_cx:
        if ex.ligand_smiles is None:
            continue
        try:
            g = smiles_to_graph_arrays(ex.ligand_smiles)
        except Exception:
            continue
        lig = to_pyg_data(g, y=ex.affinity)
        geo = _build_geo_pyg(ex)
        lb = Batch.from_data_list([lig])
        gb = Batch.from_data_list([geo])

        mean_, ep_, al_ = trained_model.predict_with_uncertainty(lb, gb, n_samples=30)
        preds.append(mean_.item())
        eps_stds.append(ep_.item())
        ale_stds.append(al_.item())
        truths.append(ex.affinity)

    n_test = len(truths)
    print(f"  Training time ({args.epochs} epochs)  : {train_sec:.1f}s")
    print(f"  Test examples evaluated  : {n_test}")
    print()

    if n_test > 0:
        try:
            ece, cal_bins = calibration_ece(preds, eps_stds, truths)
            print("--- [3/3] Calibration (empirical coverage vs nominal) ---")
            print(f"  ECE (calibration error)  : {ece:.4f}")
            print(f"  {'Nominal':>8}  {'Observed':>9}  {'Error':>7}")
            for b in cal_bins[::2]:  # every other bin for brevity
                print(f"  {b['nominal']:8.0%}  {b['observed']:9.0%}  {b['error']:7.4f}")
            print()
        except ImportError:
            print("  [skip] scipy not available for calibration curves\n")
            ece = float("nan")
            cal_bins = []

        # Fusion accuracy
        mae_fus = sum(abs(p - t) for p, t in zip(preds, truths)) / n_test
        mean_t = sum(truths) / n_test
        ss_res = sum((p - t) ** 2 for p, t in zip(preds, truths))
        ss_tot = sum((t - mean_t) ** 2 for t in truths)
        r2_fus = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Mean-predictor baseline
        mae_mean = sum(abs(mean_t - t) for t in truths) / n_test

        # GINE-only baseline
        train_smiles = [ex.ligand_smiles for ex in train_cx if ex.ligand_smiles]
        train_aff = [ex.affinity for ex in train_cx if ex.ligand_smiles]
        test_smiles = [ex.ligand_smiles for ex in test_cx if ex.ligand_smiles]
        test_aff = [ex.affinity for ex in test_cx if ex.ligand_smiles]
        mae_gine, r2_gine = train_baseline(train_smiles, train_aff, test_smiles, test_aff)

        print("--- Accuracy vs baselines ---")
        print(f"  {'Model':<25} {'MAE':>7}  {'R²':>7}")
        print(f"  {'Mean predictor':<25} {mae_mean:7.4f}  {'—':>7}")
        print(f"  {'GINE-only (2D, 20 ep)':<25} {mae_gine:7.4f}  {r2_gine:7.4f}")
        print(f"  {'FusionAffinityNet':<25} {mae_fus:7.4f}  {r2_fus:7.4f}")
        print()

        results = {
            "speed": {
                "forward_pass_batch8_ms": round(fwd_ms, 3),
                "mc_sampling_ms": {str(k): round(v, 3) for k, v in mc_times.items()},
            },
            "calibration": {
                "ece": ece if not math.isnan(ece) else None,
                "bins": cal_bins,
            },
            "accuracy": {
                "mean_predictor_mae": round(mae_mean, 4),
                "gine_only_mae": round(mae_gine, 4),
                "gine_only_r2": round(r2_gine, 4),
                "fusion_mae": round(mae_fus, 4),
                "fusion_r2": round(r2_fus, 4),
            },
        }

        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.out}")
    else:
        print("  No test examples to evaluate.")


if __name__ == "__main__":
    main()
