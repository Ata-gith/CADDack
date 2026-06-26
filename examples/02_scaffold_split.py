"""
Example 02: Scaffold-Aware Train/Test Split
============================================
Demonstrates caddack.qsar.split — Murcko scaffold splitting to avoid
scaffold leakage between train and test sets, a common pitfall in QSAR.

Requires: rdkit  (conda install -c conda-forge rdkit)
"""

import sys

import numpy as np
import pandas as pd

try:
    from caddack.qsar.split import murcko_scaffold, scaffold_split
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  conda install -c conda-forge rdkit")
    sys.exit(1)

# ── 1. Inspect Murcko scaffolds ───────────────────────────────────────────────

samples = [
    ("aspirin",      "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen",    "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("naproxen",     "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("paracetamol",  "CC(=O)Nc1ccc(O)cc1"),
    ("caffeine",     "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
]

print(f"{'Name':<15} {'Scaffold SMILES'}")
print("-" * 55)
for name, smi in samples:
    scaf = murcko_scaffold(smi)
    print(f"{name:<15} {scaf}")
print()

# ── 2. Scaffold split on a 10-molecule dataset ────────────────────────────────

np.random.seed(0)
smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",    # ibuprofen
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",  # naproxen (same scaffold as ibuprofen aryl)
    "CC(=O)Nc1ccc(O)cc1",            # paracetamol
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",    # caffeine
    "c1ccc(cc1)C(=O)O",              # benzoic acid
    "c1ccc(cc1)N",                   # aniline
    "OC(=O)c1ccccc1O",               # 2-hydroxybenzoic acid
    "c1ccc2ccccc2c1",                # naphthalene
    "c1ccc(cc1)O",                   # phenol
]
names = [
    "aspirin", "ibuprofen", "naproxen", "paracetamol", "caffeine",
    "benzoic acid", "aniline", "2-OH benzoic", "naphthalene", "phenol",
]
pic50 = np.random.uniform(4.0, 8.0, size=len(smiles_list)).round(2)

df = pd.DataFrame({
    "SMILES_canonical": smiles_list,
    "name": names,
    "pIC50": pic50,
})

train_idx, test_idx = scaffold_split(df, smiles_col="SMILES_canonical", test_size=0.2, seed=42)

print(f"Dataset size     : {len(df)}")
print(f"Train set        : {len(train_idx)} molecules  (indices {sorted(int(i) for i in train_idx)})")
print(f"Test set         : {len(test_idx)} molecules   (indices {sorted(int(i) for i in test_idx)})")
print()
print("Train molecules:")
for i in sorted(train_idx):
    print(f"  [{i}] {df.loc[i,'name']:<15}  pIC50={df.loc[i,'pIC50']}")
print()
print("Test molecules:")
for i in sorted(test_idx):
    print(f"  [{i}] {df.loc[i,'name']:<15}  pIC50={df.loc[i,'pIC50']}")
print()

# ── 3. Why scaffold split matters ────────────────────────────────────────────

print("Scaffold split groups structurally similar molecules together,")
print("preventing the model from memorising scaffold patterns during training")
print("and giving a realistic estimate of generalisation to new chemotypes.")
