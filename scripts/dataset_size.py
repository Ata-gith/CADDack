#!/usr/bin/env python3
"""What does PDBbind's general set actually buy: accuracy, or coverage?

Groups the test set by the BASELINE arm's applicability domain, so both arms are
scored on the same molecules, and separately counts how many test ligands the
extra data pulls inside the domain.
"""
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from caddack.qsar import canonicalize_smiles, parse_smiles
from caddack.qsar.applicability import SimilarityAD
from caddack.qsar.descriptors import mol_to_ecfp_array
from caddack.qsar.split import murcko_scaffold, scaffold_split

CACHE = sys.argv[1] if len(sys.argv) > 1 else "affinity_features_general_raw.npz"
PICKLE = sys.argv[2] if len(sys.argv) > 2 else "pdbbind_examples.pkl"
blob = np.load(CACHE, allow_pickle=True)
X, y = blob["X"], blob["y"]
smiles = list(blob["smiles"])
ids = list(blob["ids"])
ex = pickle.load(open(PICKLE, "rb"))
refined_ids = {e.pdb_id for e in ex if getattr(e, "is_refined", True)}
is_ref = np.array([pid in refined_ids for pid in ids])

FP = {}
def fp(i):
    if i not in FP:
        m = parse_smiles(smiles[i])
        FP[i] = None if m is None else np.unpackbits(mol_to_ecfp_array(m, n_bits=1024))
    return FP[i]

def key(s):
    if not s:
        return None
    sc = murcko_scaffold(s)
    return sc if sc else (canonicalize_smiles(s) or s)
keys = np.array([key(s) for s in smiles], dtype=object)

ref_pos = np.flatnonzero(is_ref)
ref_smiles = [smiles[i] for i in ref_pos]
gen_pos = np.flatnonzero(~is_ref)

res = []
for seed in range(5):
    tr_r, te_r = scaffold_split(pd.DataFrame({"smiles": ref_smiles}),
                                smiles_col="smiles", test_size=0.2, seed=seed)
    tr, te = ref_pos[tr_r], ref_pos[te_r]
    test_keys = set(keys[te]) - {None}
    extra = np.array([i for i in gen_pos if keys[i] is not None and keys[i] not in test_keys])
    big = np.concatenate([tr, extra])

    te_fp = np.stack([fp(i) for i in te])
    ad_small = SimilarityAD.fit(np.stack([f for f in (fp(i) for i in tr) if f is not None]),
                                percentile=5.0)
    ad_big = SimilarityAD.fit(np.stack([f for f in (fp(i) for i in big) if f is not None]),
                              percentile=5.0)
    sim_small = np.array([ad_small.similarity(f[None, :])[0] for f in te_fp])
    sim_big = np.array([ad_big.similarity(f[None, :])[0] for f in te_fp])

    preds = {}
    for name, idx in (("small", tr), ("big", big)):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X[idx], y[idx])
        preds[name] = rf.predict(X[te])
    e_small = np.abs(preds["small"] - y[te])
    e_big = np.abs(preds["big"] - y[te])

    out_small = sim_small < ad_small.threshold          # grouping fixed by the baseline
    moved = out_small & (sim_big >= ad_big.threshold)   # pulled inside by the extra data
    res.append({
        "n_out_small": int(out_small.sum()),
        "n_out_big": int((sim_big < ad_big.threshold).sum()),
        "n_moved": int(moved.sum()),
        "in_small": float(e_small[~out_small].mean()), "in_big": float(e_big[~out_small].mean()),
        "out_small": float(e_small[out_small].mean()), "out_big": float(e_big[out_small].mean()),
        "moved_small": float(e_small[moved].mean()) if moved.any() else np.nan,
        "moved_big": float(e_big[moved].mean()) if moved.any() else np.nan,
        "nn_small": float(sim_small.mean()), "nn_big": float(sim_big.mean()),
    })
    print(f"  seed {seed}: outside {res[-1]['n_out_small']} -> {res[-1]['n_out_big']}, "
          f"{res[-1]['n_moved']} pulled inside", flush=True)

print("\nSame molecules in both arms, grouped by the BASELINE model's domain:")
for label, a, b in (("in-domain MAE", "in_small", "in_big"),
                    ("out-of-domain MAE", "out_small", "out_big"),
                    ("MAE on the ones pulled inside", "moved_small", "moved_big")):
    da = np.array([r[a] for r in res])
    db = np.array([r[b] for r in res])
    d = db - da
    t = stats.ttest_1samp(d[~np.isnan(d)], 0)
    print(f"  {label:<32} {da.mean():.3f} -> {db.mean():.3f}   "
          f"change {d.mean():+.3f}   improved {(d<0).sum()}/5   p = {t.pvalue:.3f}")

print("\nCoverage:")
print(f"  test ligands outside the domain  {np.mean([r['n_out_small'] for r in res]):.0f}"
      f" -> {np.mean([r['n_out_big'] for r in res]):.0f} of 740")
print(f"  mean nearest-training similarity {np.mean([r['nn_small'] for r in res]):.3f}"
      f" -> {np.mean([r['nn_big'] for r in res]):.3f}")
