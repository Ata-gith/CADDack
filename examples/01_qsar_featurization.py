"""
Example 01: QSAR Featurization
================================
Demonstrates caddack.qsar — converting SMILES strings into physicochemical
descriptors and ECFP fingerprint bits, ready for classical ML.

Requires: rdkit  (conda install -c conda-forge rdkit)
"""

import sys

try:
    import pandas as pd
    from caddack.qsar.descriptors import (
        parse_smiles,
        canonicalize_smiles,
        strip_salts,
        smiles_to_mol_clean,
        mol_to_basic_features,
        mol_to_ecfp_bits,
        smiles_to_features,
        featurize_dataframe,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  conda install -c conda-forge rdkit")
    sys.exit(1)

# ── 1. Basic SMILES utilities ────────────────────────────────────────────────

noisy_smiles = "CC(=O)Oc1ccccc1C(=O)O.[Na+]"  # aspirin + sodium counter-ion

canonical = canonicalize_smiles(noisy_smiles)
print(f"Input SMILES     : {noisy_smiles}")
print(f"Canonical        : {canonical}")
print(f"Salt stripped    : {strip_salts(noisy_smiles)}")
print()

# ── 2. Physicochemical descriptors for one molecule ──────────────────────────

mol = smiles_to_mol_clean(noisy_smiles)
physchem = mol_to_basic_features(mol)
print("Physicochemical descriptors:")
for name, val in physchem.items():
    print(f"  {name:20s} {val:.3f}")
print()

# ── 3. ECFP2 fingerprint (2048 bits) ─────────────────────────────────────────

ecfp = mol_to_ecfp_bits(mol, radius=2, n_bits=2048)
on_bits = [k for k, v in ecfp.items() if v == 1]
print(f"ECFP2 bits set   : {len(on_bits)} / 2048")
print(f"First 5 on bits  : {on_bits[:5]}")
print()

# ── 4. All features in one call ──────────────────────────────────────────────

feats = smiles_to_features("CC(=O)Nc1ccc(O)cc1", radius=2, n_bits=2048)  # paracetamol
print("smiles_to_features keys (first 10):")
print(" ", list(feats.keys())[:10], "...")
print()

# ── 5. Featurize a DataFrame ─────────────────────────────────────────────────

molecules = pd.DataFrame({
    "SMILES": [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # caffeine
    ],
    "name": ["aspirin", "ibuprofen", "caffeine"],
    "pIC50": [5.2, 6.1, 4.8],
})

print("Featurizing DataFrame with 3 molecules ...")
feat_df = featurize_dataframe(molecules, smiles_col="SMILES", target_col="pIC50")
print(f"Output shape     : {feat_df.shape}")
print(f"Physicochemical columns: {[c for c in feat_df.columns if not c.startswith('ECFP')][:8]}")
print()
print(feat_df[["name", "pIC50", "MolWt", "LogP", "TPSA", "NumHBD", "NumHBA"]].to_string(index=False))
