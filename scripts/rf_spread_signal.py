#!/usr/bin/env python3
r"""Does the forest's tree spread rank its own errors, across scaffold splits?

The network's Bayesian epistemic term is negative on every seed (see
docs/README.md, "Sampling the encoders"). The forest has an uncertainty
estimate that costs nothing -- the standard deviation across its trees, a
by-product of bagging -- and this measures whether it does the job the
Bayesian machinery could not. It does: rho +0.21 against the errors, about
2.7x the model-free distance lookup, and positive on every split.

    python scripts/rf_spread_signal.py [features.npz]

Expects the feature cache train_affinity_rf.py writes (X, y, mask, smiles).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from caddack.qsar import parse_smiles
from caddack.qsar.applicability import SimilarityAD
from caddack.qsar.descriptors import mol_to_ecfp_array
from caddack.qsar.split import scaffold_split

CACHE = sys.argv[1] if len(sys.argv) > 1 else "affinity_features_refined_raw.npz"
blob = np.load(CACHE, allow_pickle=True)
X, y, mask = blob["X"], blob["y"], blob["mask"]
smiles = list(blob["smiles"])
print("X", X.shape, "mask", mask.shape, "valid", mask.sum())

def fp(s):
    m = parse_smiles(s)
    return np.unpackbits(mol_to_ecfp_array(m, n_bits=1024)) if m is not None else None

position = np.cumsum(mask) - 1
# The split runs over the full example list; reconstruct it from the valid rows'
# smiles placed back at their dataset positions.
full = np.empty(len(mask), dtype=object)
full[mask] = smiles
full[~mask] = ""
print(f"{'seed':<6}{'r':>8}{'MAE':>8}{'rho(spread,err)':>18}{'rho(dist,err)':>16}"
      f"{'in MAE':>9}{'out MAE':>9}{'cov1x':>8}{'cov2x':>8}")
rows = []
for seed in range(5):
    tr_all, te_all = scaffold_split(pd.DataFrame({"smiles": full}),
                                    smiles_col="smiles", test_size=0.2, seed=seed)
    tr = position[[i for i in tr_all if mask[i]]]
    te = position[[i for i in te_all if mask[i]]]
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    per_tree = np.stack([t.predict(X[te]) for t in rf.estimators_])
    pred, spread = per_tree.mean(0), per_tree.std(0, ddof=1)
    err = np.abs(pred - y[te])
    tr_fp = [f for f in (fp(smiles[i]) for i in tr) if f is not None]
    ad = SimilarityAD.fit(np.stack(tr_fp), percentile=5.0)
    sims = np.array([ad.similarity(fp(smiles[i])[None, :])[0] if fp(smiles[i]) is not None
                     else np.nan for i in te])
    ok = ~np.isnan(sims)
    inside, outside = sims >= ad.threshold, sims < ad.threshold
    r = {"seed": seed,
         "r": float(np.corrcoef(pred, y[te])[0, 1]), "mae": float(err.mean()),
         "rho_spread": spearmanr(spread, err).statistic,
         "rho_dist": spearmanr(-sims[ok], err[ok]).statistic,
         "in_mae": float(err[inside].mean()), "out_mae": float(err[outside].mean()),
         "cov1": float((err <= spread).mean()), "cov2": float((err <= 2*spread).mean())}
    rows.append(r)
    print(f"{seed:<6}{r['r']:>8.3f}{r['mae']:>8.3f}{r['rho_spread']:>18.3f}"
          f"{r['rho_dist']:>16.3f}{r['in_mae']:>9.2f}{r['out_mae']:>9.2f}"
          f"{r['cov1']*100:>7.0f}%{r['cov2']*100:>7.0f}%")
def agg(k):
    col = [r[k] for r in rows]
    return np.mean(col), np.std(col, ddof=1)


print()
for k in ("rho_spread", "rho_dist", "in_mae", "out_mae", "cov1", "cov2"):
    m, sd = agg(k)
    print(f"  {k:<12} {m:+.3f} +- {sd:.3f}")
