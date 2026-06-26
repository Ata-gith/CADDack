"""
Example 04: Train a GNN from a CSV File
=========================================
Demonstrates caddack.gnn.train.train_from_csv — the high-level entry
point that reads a SMILES CSV and trains a GCN or GINE end-to-end.

Two variants are shown:
  - GINE classification (active/inactive label)
  - GCN regression (continuous pIC50)

Requires: rdkit + torch + torch-geometric + scikit-learn
  conda install -c conda-forge rdkit
  pip install torch torch-geometric scikit-learn
"""

import sys
import json
import tempfile
from pathlib import Path

import pandas as pd

try:
    from caddack.gnn.train import train_from_csv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:")
    print("  conda install -c conda-forge rdkit")
    print("  pip install torch torch-geometric scikit-learn")
    sys.exit(1)

# ── Inline dataset (15 drug-like molecules) ───────────────────────────────────

DATA = pd.DataFrame({
    "smiles": [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC(=O)Nc1ccc(O)cc1",
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "c1ccc(cc1)C(=O)O",
        "OC(=O)c1ccccc1O",
        "c1ccc(cc1)N",
        "c1ccc2ccccc2c1",
        "c1ccc(cc1)O",
        "CC(=O)c1ccccc1",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "CC1=CC(=O)c2ccccc2C1=O",
        "O=C(O)c1ccc(N)cc1",
        "CC(=O)Nc1ccc(cc1)S(N)(=O)=O",
        "O=C(O)/C=C/c1ccccc1",
    ],
    "pIC50": [5.2, 6.1, 4.8, 3.9, 4.3, 4.5, 3.1, 3.8, 3.6, 5.0,
              6.4, 5.7, 4.1, 6.8, 4.9],
    "active": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
})

# ── Variant 1: GINE classification ───────────────────────────────────────────

print("=" * 55)
print("Variant 1: GINE classification (active / inactive)")
print("=" * 55)

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    DATA.to_csv(f, index=False)
    csv_path = f.name

with tempfile.TemporaryDirectory() as outdir:
    metrics = train_from_csv(
        csv_path=csv_path,
        smiles_col="smiles",
        target_col="active",
        outdir=outdir,
        model_name="gine",
        task="classification",
        epochs=20,
        batch_size=8,
        lr=1e-3,
        hidden_channels=64,
        num_layers=2,
        test_size=0.25,
        seed=42,
    )
    print("Metrics:", json.dumps(metrics, indent=2))
    saved = list(Path(outdir).iterdir())
    print(f"Saved files: {[f.name for f in saved]}")

print()

# ── Variant 2: GCN regression ─────────────────────────────────────────────────

print("=" * 55)
print("Variant 2: GCN regression (pIC50)")
print("=" * 55)

with tempfile.TemporaryDirectory() as outdir:
    metrics = train_from_csv(
        csv_path=csv_path,
        smiles_col="smiles",
        target_col="pIC50",
        outdir=outdir,
        model_name="gcn",
        task="regression",
        epochs=20,
        batch_size=8,
        lr=1e-3,
        hidden_channels=64,
        num_layers=2,
        test_size=0.25,
        seed=42,
    )
    print("Metrics:", json.dumps(metrics, indent=2))

Path(csv_path).unlink(missing_ok=True)
print()
print("train_from_csv saves model.pt + metrics.json to outdir.")
print("Set epochs=100+ and a larger dataset for meaningful results.")
