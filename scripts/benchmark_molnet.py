#!/usr/bin/env python3
"""Benchmark CADDack's 2D GNNs (GCN / GINE) on standard MoleculeNet datasets.

Datasets (plain SMILES CSVs from the DeepChem S3 mirror):
  - ESOL      (regression)     : aqueous solubility, 1128 mols
  - FreeSolv  (regression)     : hydration free energy, 642 mols
  - BBBP      (classification) : blood-brain-barrier penetration, ~2039 mols
  - BACE      (classification) : beta-secretase inhibition, 1513 mols

Runs ``caddack.gnn.train.train_from_csv`` end-to-end (random 80/20 split) and
prints a summary table. Unparseable SMILES are skipped by the loader itself.

Usage:
    python scripts/benchmark_molnet.py --download --data-dir /tmp/molnet \\
        --outdir /tmp/molnet_runs --epochs 40 --model gine
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from caddack.gnn.train import train_from_csv


# name -> (download url, smiles_col, target_col, task)
DATASETS = {
    "ESOL": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smiles", "measured log solubility in mols per litre", "regression",
    ),
    "FreeSolv": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles", "expt", "regression",
    ),
    "BBBP": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles", "p_np", "classification",
    ),
    "BACE": (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "mol", "Class", "classification",
    ),
}


def _fname(name: str) -> str:
    return f"{name.lower()}.csv"


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, *_rest) in DATASETS.items():
        dest = data_dir / _fname(name)
        if dest.exists():
            print(f"  [have] {dest}")
            continue
        print(f"  [get ] {name} <- {url}")
        urllib.request.urlretrieve(url, dest)


def run(data_dir: Path, outdir: Path, epochs: int, model: str) -> list[dict]:
    results = []
    for name, (_url, smiles_col, target_col, task) in DATASETS.items():
        csv_path = data_dir / _fname(name)
        if not csv_path.exists():
            print(f"[skip] {name}: {csv_path} not found (run with --download)")
            continue

        # Normalise to a clean two-column CSV; the loader skips bad SMILES itself.
        df = pd.read_csv(csv_path)
        clean = pd.DataFrame({
            "smiles": df[smiles_col],
            "target": pd.to_numeric(df[target_col], errors="coerce"),
        }).dropna()
        clean_path = outdir / f"{name.lower()}_clean.csv"
        clean.to_csv(clean_path, index=False)

        print(f"\n=== {name} ({task}, {len(clean)} molecules, model={model}) ===")
        t0 = time.perf_counter()
        metrics = train_from_csv(
            csv_path=str(clean_path),
            smiles_col="smiles",
            target_col="target",
            outdir=str(outdir / name.lower()),
            model_name=model,
            task=task,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            hidden_channels=128,
            num_layers=3,
            test_size=0.2,
            seed=42,
        )
        elapsed = time.perf_counter() - t0
        results.append({"dataset": name, "task": task, "n": len(clean),
                        "model": model, "epochs": epochs,
                        "seconds": round(elapsed, 1), **metrics})
        print(f"  metrics: {json.dumps(metrics)}  ({elapsed:.1f}s)")
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True, help="where dataset CSVs live")
    ap.add_argument("--outdir", required=True, help="where to write models/results")
    ap.add_argument("--download", action="store_true",
                    help="download missing datasets before running")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model", default="gine", choices=["gine", "gcn"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("Downloading datasets ...")
        download(data_dir)

    results = run(data_dir, outdir, args.epochs, args.model)

    print("\n\n================ SUMMARY ================")
    print(f"{'Dataset':<10} {'Task':<15} {'N':>5} {'metric1':>18} {'metric2':>12} {'time':>7}")
    for r in results:
        if r["task"] == "regression":
            m1 = f"MAE={r.get('mae', float('nan')):.4f}"
            m2 = f"R2={r.get('r2', float('nan')):.4f}"
        else:
            m1 = f"AUC={r.get('auc_roc', float('nan')):.4f}"
            m2 = f"Acc={r.get('accuracy', float('nan')):.4f}"
        print(f"{r['dataset']:<10} {r['task']:<15} {r['n']:>5} {m1:>18} {m2:>12} {r['seconds']:>6.0f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
