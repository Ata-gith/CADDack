#!/usr/bin/env python3
"""Rank-correlate every uncertainty signal against the error, per training run.

Reads the bundles that ``build_fusion_demo.py`` writes and reports rho against
|error| for the epistemic term, the total, and the model-free distance to
training chemistry, then aggregates by arm. Used to test whether sampling the
encoders as well as the head rescues the epistemic signal; it does not. See
"Sampling the encoders, and why it did not help" in docs/README.md.

    for seed in 0 1; do
      for arm in head towers; do
        flag=""; [ "$arm" = towers ] && flag="--bayesian-towers"
        python scripts/build_fusion_demo.py --data-dir /path/to/pdbbind \\
          --vina-features --geo-readout sum $flag --epochs 12 \\
          --seed $seed --init-seed $((100+seed)) \\
          --prefix "${arm}_s${seed}" --out "out/${arm}_s${seed}.json"
      done
    done
    python scripts/bayes_tower_rho.py out/

Bundle names must start with the arm, as ``<arm>_s<seed>.json``.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

rows = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_s*.json"))):
    tag = os.path.basename(path)[:-5]
    b = json.load(open(path))
    c = b["complexes"]
    err = np.array([abs(r["pred"] - r["measured"]) for r in c])
    epi = np.array([r["epi"] for r in c])
    ale = np.array([r["ale"] for r in c])
    tot = np.sqrt(epi**2 + ale**2)
    sim = np.array([r["ad_sim"] if r["ad_sim"] is not None else np.nan for r in c])
    ok = ~np.isnan(sim)
    rows[tag] = {
        "pearson": b["metrics"]["pearson"],
        "mae": b["metrics"]["mae"],
        "spread": b["metrics"]["prediction_spread_ratio"],
        "rho_epi": spearmanr(epi, err).statistic,
        "rho_tot": spearmanr(tot, err).statistic,
        "rho_dist": spearmanr(-sim[ok], err[ok]).statistic,
        "epi_share": float(np.mean(epi**2 / (epi**2 + ale**2))),
    }

hdr = ("run", "pearson", "MAE", "spread", "rho(epi,err)", "rho(total,err)",
       "rho(dist,err)", "epi var share")
print(f"{hdr[0]:<11}" + "".join(f"{h:>15}" for h in hdr[1:]))
for tag, m in rows.items():
    print(f"{tag:<11}" + "".join(f"{m[k]:>15.3f}" for k in
          ("pearson","mae","spread","rho_epi","rho_tot","rho_dist","epi_share")))

for arm in ("head", "towers"):
    sub = [m for t, m in rows.items() if t.startswith(arm)]
    if len(sub) < 2:
        continue
    print(f"\n{arm}: " + "  ".join(
        f"{k}={np.mean([s[k] for s in sub]):+.3f}+-{np.std([s[k] for s in sub], ddof=1):.3f}"
        for k in ("rho_epi", "pearson", "mae")))
