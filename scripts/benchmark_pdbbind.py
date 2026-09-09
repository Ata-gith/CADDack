#!/usr/bin/env python3
"""Benchmark FusionAffinityNet on real PDBbind protein-ligand complexes.

Downloads PDBbind v2015 from the DeepChem public mirror, builds ComplexExamples
from each complex's pocket.pdb + ligand.mol2, and trains the two-tower Bayesian
affinity model.

Splits:
  core     -- train on refined-minus-core, test on the core set (CASF-style).
              Optimistic: refined and core share many scaffolds.
  scaffold -- hold out 20% of Murcko scaffolds. Harder and more realistic:
              test ligands are chemically novel.

Usage:
    # download + build the dataset cache (~2GB download, once)
    python scripts/benchmark_pdbbind.py --data-dir /tmp/pdbbind --download --build

    # CASF-style benchmark on the refined set
    python scripts/benchmark_pdbbind.py --data-dir /tmp/pdbbind \\
        --refined-only --split core --epochs 15 --prefix refined

    # harder scaffold-split benchmark
    python scripts/benchmark_pdbbind.py --data-dir /tmp/pdbbind \\
        --refined-only --split scaffold --epochs 20 --prefix scaffold

Training checkpoints every epoch to <prefix>_ckpt.pt and resumes automatically,
so an interrupted run can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TARBALL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz"
TARBALL_NAME = "pdbbind_v2015.tar.gz"
CACHE_NAME = "pdbbind_examples.pkl"


# ---------------------------------------------------------------- download ---
def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / TARBALL_NAME
    if dest.exists():
        print(f"  [have] {dest} ({dest.stat().st_size/1e9:.2f} GB)")
        return
    print(f"  [get ] {TARBALL_URL}\n         -> {dest} (~2 GB, may take a few minutes)")
    urllib.request.urlretrieve(TARBALL_URL, dest)


# ------------------------------------------------------------------- build ---
def _parse_index(path: Path) -> dict[str, float]:
    """PDBbind INDEX file: col 0 = PDB id, col 3 = -logKd/Ki."""
    out: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    out[parts[0]] = float(parts[3])
                except ValueError:
                    pass
    return out


def build(data_dir: Path) -> None:
    """Extract pocket+ligand files and cache parsed ComplexExamples."""
    from caddack.gnn.geometry import load_complex

    tar = data_dir / TARBALL_NAME
    if not tar.exists():
        raise SystemExit(f"{tar} not found; run with --download first")

    subprocess.run(
        ["tar", "xzf", TARBALL_NAME,
         "v2015/INDEX_general_PL_data.2015", "v2015/INDEX_refined_data.2015"],
        cwd=data_dir, check=True,
    )
    general = _parse_index(data_dir / "v2015/INDEX_general_PL_data.2015")
    refined_ids = set(_parse_index(data_dir / "v2015/INDEX_refined_data.2015"))
    print(f"  index: {len(general)} general / {len(refined_ids)} refined")

    filelist = data_dir / "filelist.txt"
    with open(filelist, "w") as fl:
        for pid in general:
            fl.write(f"v2015/{pid}/{pid}_pocket.pdb\n")
            fl.write(f"v2015/{pid}/{pid}_ligand.mol2\n")
    t0 = time.perf_counter()
    # missing members are tolerated: the general index lists a few absent dirs
    subprocess.run(["tar", "xzf", TARBALL_NAME, "-T", "filelist.txt"],
                   cwd=data_dir, check=False)
    print(f"  extracted in {time.perf_counter()-t0:.0f}s")

    examples, n_seen, t0 = [], 0, time.perf_counter()
    for pid, aff in general.items():
        pocket = data_dir / f"v2015/{pid}/{pid}_pocket.pdb"
        ligand = data_dir / f"v2015/{pid}/{pid}_ligand.mol2"
        if not (pocket.exists() and ligand.exists()):
            continue
        n_seen += 1
        ex = load_complex(str(pocket), str(ligand), affinity=float(aff),
                          pdb_id=pid, cutoff=6.0)
        if ex is None or not ex.ligand_smiles:
            continue
        ex.is_refined = pid in refined_ids
        examples.append(ex)
        if n_seen % 2000 == 0:
            print(f"    ...{n_seen} seen, {len(examples)} valid "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    aff = np.array([e.affinity for e in examples])
    n_ref = sum(getattr(e, "is_refined", False) for e in examples)
    print(f"  built {len(examples)}/{n_seen} valid ({n_ref} refined) "
          f"in {time.perf_counter()-t0:.0f}s")
    print(f"  affinity(-logKd/Ki): min={aff.min():.2f} max={aff.max():.2f} "
          f"mean={aff.mean():.2f} std={aff.std():.2f}")
    with open(data_dir / CACHE_NAME, "wb") as f:
        pickle.dump(examples, f)
    print(f"  cached -> {data_dir / CACHE_NAME}")


# ------------------------------------------------------------------- train ---
def run(args) -> dict:
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

    from caddack.gnn.bayes import elbo_loss
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet
    from caddack.gnn.train import _build_geo_pyg
    from caddack.qsar.split import scaffold_split

    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    with open(data_dir / CACHE_NAME, "rb") as f:
        examples = pickle.load(f)

    if args.refined_only:
        examples = [e for e in examples if getattr(e, "is_refined", True)]
        print(f"  refined-only -> {len(examples)} complexes")

    if args.split == "core":
        core = pd.read_csv(args.core_csv, usecols=["pdb_id"])["pdb_id"].astype(str)
        core_ids = set(core)
        train_ex = [e for e in examples if e.pdb_id not in core_ids]
        test_ex = [e for e in examples if e.pdb_id in core_ids]
    else:
        sdf = pd.DataFrame({"smiles": [e.ligand_smiles for e in examples]})
        tr, te = scaffold_split(sdf, smiles_col="smiles",
                                test_size=args.test_size, seed=args.seed)
        train_ex = [examples[i] for i in tr]
        test_ex = [examples[i] for i in te]

    if args.subsample and len(train_ex) > args.subsample:
        rng = np.random.RandomState(args.seed)
        train_ex = [train_ex[i] for i in
                    rng.choice(len(train_ex), args.subsample, replace=False)]
    print(f"  split={args.split}  train={len(train_ex)}  test={len(test_ex)}")

    # Standardise the target with TRAIN statistics only (never test — that leaks).
    # Without this the Bayesian head — regularised toward a zero-mean prior — leaves
    # a systematic offset in the predictions (~1 pK on the scaffold split), which
    # destroys R2 even when the ranking is good.
    train_mean = float(np.mean([e.affinity for e in train_ex]))
    if args.standardize_target:
        y_mean = train_mean
        y_std = float(np.std([e.affinity for e in train_ex], ddof=1)) or 1.0
    else:
        y_mean, y_std = 0.0, 1.0
    print(f"  target standardisation "
          f"{'on' if args.standardize_target else 'off'}: "
          f"mean={y_mean:.3f} std={y_std:.3f}")

    def make_pair(ex):
        lig = to_pyg_data(smiles_to_graph_arrays(ex.ligand_smiles), y=ex.affinity)
        geo = _build_geo_pyg(ex)
        geo.y = torch.tensor([(float(ex.affinity) - y_mean) / y_std], dtype=torch.float)
        return lig, geo

    def build_pairs(exs):
        out = []
        for ex in exs:
            try:
                out.append(make_pair(ex))
            except Exception:
                pass  # unparseable ligand SMILES; already rare after the cache step
        return out

    def collate(batch):
        ligs, geos = zip(*batch)
        return Batch.from_data_list(list(ligs)), Batch.from_data_list(list(geos))

    train_pairs, test_pairs = build_pairs(train_ex), build_pairs(test_ex)
    n_train = len(train_pairs)
    train_loader = DataLoader(train_pairs, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_pairs, batch_size=64, shuffle=False,
                             collate_fn=collate)

    model = FusionAffinityNet.build(
        ligand_in_channels=7, ligand_edge_dim=7,
        hidden_channels=args.hidden, num_gine_layers=3, num_geo_interactions=2,
        num_rbf=24, cutoff=6.0, bayesian_hidden=[256, 128],
        prior_sigma=args.prior_sigma, dropout=args.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path, best_path = f"{args.prefix}_ckpt.pt", f"{args.prefix}_best.pt"
    start_epoch, best_loss = 0, float("inf")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch, best_loss = state["epoch"], state.get("best_loss", float("inf"))
        print(f"  resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        beta = args.kl_weight * min(1.0, (epoch + 1) / max(args.kl_warmup, 1))
        for lig_b, geo_b in train_loader:
            opt.zero_grad()
            mu, log_var = model(lig_b, geo_b)
            loss = elbo_loss(mu, log_var, geo_b.y.view(-1), model.kl(),
                             n_train=n_train, kl_weight=beta)
            loss.backward()
            # Bayesian layers can spike without clipping; 5.0 keeps training stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss={avg:.4f}  beta={beta:.2f}"
              f"  ({time.perf_counter()-t0:.0f}s)", flush=True)
        if beta >= args.kl_weight and avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "best_loss": best_loss}, ckpt_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()
    preds, eps, ale, truth = [], [], [], []
    with torch.no_grad():
        for lig_b, geo_b in test_loader:
            m, e, a = model.predict_with_uncertainty(lig_b, geo_b,
                                                     n_samples=args.mc_samples)
            preds += m.tolist(); eps += e.tolist(); ale += a.tolist()
            truth += geo_b.y.view(-1).tolist()
    # rescale back to affinity units (uncertainties are scale-only, no offset)
    preds = np.array(preds) * y_std + y_mean
    truth = np.array(truth) * y_std + y_mean
    eps, ale = np.array(eps) * y_std, np.array(ale) * y_std
    total_std = np.sqrt(eps ** 2 + ale ** 2)
    metrics = {
        "split": args.split,
        "n_train": n_train,
        "n_test": len(preds),
        "mae": float(np.mean(np.abs(preds - truth))),
        "rmse": float(np.sqrt(np.mean((preds - truth) ** 2))),
        "r2": float(1 - np.sum((preds - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "pearson": float(np.corrcoef(preds, truth)[0, 1]),
        "baseline_mae": float(np.mean(np.abs(train_mean - truth))),
        "coverage_1sigma": float(np.mean(np.abs(preds - truth) <= total_std)),
        "coverage_2sigma": float(np.mean(np.abs(preds - truth) <= 2 * total_std)),
    }
    print("\n  " + json.dumps(metrics, indent=2).replace("\n", "\n  "))
    with open(f"{args.prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="where the dataset lives")
    ap.add_argument("--download", action="store_true", help="fetch the tarball")
    ap.add_argument("--build", action="store_true", help="extract + cache examples")
    ap.add_argument("--core-csv", default=None,
                    help="pdbbind_core_df.csv.gz (for --split core)")
    ap.add_argument("--split", choices=["core", "scaffold"], default="core")
    ap.add_argument("--refined-only", action="store_true")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-warmup", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--no-standardize-target", dest="standardize_target",
                    action="store_false",
                    help="train on raw affinities instead of z-scored targets")
    ap.set_defaults(standardize_target=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="pdbbind")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    data_dir = Path(args.data_dir)
    if args.download:
        print("Downloading PDBbind ...")
        download(data_dir)
    if args.build:
        print("Building dataset cache ...")
        build(data_dir)
    if not (data_dir / CACHE_NAME).exists():
        raise SystemExit("No dataset cache; run with --download --build first")
    if args.split == "core" and not args.core_csv:
        raise SystemExit("--split core requires --core-csv (pdbbind_core_df.csv.gz)")
    run(args)


if __name__ == "__main__":
    main()
