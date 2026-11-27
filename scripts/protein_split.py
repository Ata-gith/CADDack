#!/usr/bin/env python3
"""Does holding out whole PROTEINS, not just scaffolds, change the result?

    python scripts/protein_split.py [jaccard-threshold]   # default 0.7


The page has always said scaffold splitting separates ligands but not proteins,
so the same pocket can sit on both sides. This measures what that is worth.
Protein groups come from pocket-residue identity (see pocket_groups.py); the
threshold is set where chance cannot reach: at 0.7 only 0.01% of random pairs
link, at 0.5 only 0.39%. Both thresholds are reported in docs/README.md, and the
conclusion is the same at each.

The arm that matters is the last one. Enforcing disjointness removes training
rows, and less data alone lowers the score, so a same-size random-drop control
runs alongside: the difference between "both disjoint" and that control is the
leakage, net of the data it costs to remove.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caddack.qsar import canonicalize_smiles
from caddack.qsar.split import murcko_scaffold, scaffold_split

ROOT = Path("/home/user/pdbbind/v2015")
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7

def pocket_residues(pid):
    f = ROOT / pid / f"{pid}_pocket.pdb"
    if not f.exists():
        return None
    return {(line[17:20].strip(), line[22:26].strip())
            for line in f.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 27} or None

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0

blob = np.load("/home/user/pdbbind/affinity_features_refined_raw.npz", allow_pickle=True)
X, y = blob["X"], blob["y"]
ids = list(blob["ids"])
smiles = list(blob["smiles"])
print(f"X {X.shape}")

res = {p: r for p in ids if (r := pocket_residues(p)) is not None}
parent = {p: p for p in ids}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
index = {}
for k, v in res.items():
    for r in v:
        index.setdefault(r, []).append(k)
seen = set()
for _res_key, members in index.items():
    if len(members) <= 60:
        for i, a in enumerate(members):
            for b in members[i+1:]:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                if jaccard(res[a], res[b]) >= THRESH:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
prot = np.array([find(p) for p in ids], dtype=object)
sizes = pd.Series(prot).value_counts()
print(f"protein groups at Jaccard >= {THRESH}: {sizes.size} for {len(ids)} complexes; "
      f"largest {list(sizes.values[:8])}; in a group of 2+: "
      f"{int(sizes[sizes>1].sum())} ({sizes[sizes>1].sum()/len(ids)*100:.0f}%)")

def scaf(s):
    if not s:
        return None
    sc = murcko_scaffold(s)
    return sc if sc else (canonicalize_smiles(s) or s)
scaff = np.array([scaf(s) for s in smiles], dtype=object)

def group_split(groups, seed, test_frac=0.2):
    """Hold out whole groups until the test set is big enough (smallest first)."""
    rng = random.Random(seed)
    uniq = list(pd.unique(groups))
    rng.shuffle(uniq)
    target, test_groups, n = test_frac * len(groups), set(), 0
    for g in uniq:
        if n >= target:
            break
        test_groups.add(g)
        n += int((groups == g).sum())
    te = np.flatnonzero(np.isin(groups, list(test_groups)))
    tr = np.flatnonzero(~np.isin(groups, list(test_groups)))
    return tr, te

def run(tr, te, seed):
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    p = rf.predict(X[te])
    return float(np.corrcoef(p, y[te])[0, 1]), float(np.mean(np.abs(p - y[te]))), len(te)

arms = {"scaffold split (as shipped)": [], "protein split": [], "both disjoint": [],
        "control: same n, random drop": []}
for seed in range(5):
    tr_s, te_s = scaffold_split(pd.DataFrame({"smiles": smiles}), smiles_col="smiles",
                                test_size=0.2, seed=seed)
    arms["scaffold split (as shipped)"].append(run(np.asarray(tr_s), np.asarray(te_s), seed))

    tr_p, te_p = group_split(prot, seed)
    arms["protein split"].append(run(tr_p, te_p, seed))

    # strictest: test proteins held out AND training scaffolds that appear in test removed
    test_scaf = set(scaff[te_p]) - {None}
    tr_b = np.array([i for i in tr_p if scaff[i] not in test_scaf])
    arms["both disjoint"].append(run(tr_b, te_p, seed))
    rng = np.random.default_rng(seed)
    tr_c = rng.choice(tr_p, size=len(tr_b), replace=False)
    arms["control: same n, random drop"].append(run(tr_c, te_p, seed))
    print(f"  seed {seed}: scaffold test {len(te_s)}, protein test {len(te_p)}, "
          f"both-train {len(tr_b)} of {len(tr_p)}", flush=True)

print(f"\n{'split':<30}{'Pearson r':>18}{'MAE':>16}{'test n':>9}")
for name, v in arms.items():
    r = np.array([a for a, _, _ in v])
    m = np.array([b for _, b, _ in v])
    n = int(np.mean([c for _, _, c in v]))
    print(f"{name:<30}{r.mean():>11.3f} +-{r.std(ddof=1):.3f}"
          f"{m.mean():>10.3f} +-{m.std(ddof=1):.3f}{n:>9}")
a = np.array([x[0] for x in arms["scaffold split (as shipped)"]])
b = np.array([x[0] for x in arms["protein split"]])
c = np.array([x[0] for x in arms["both disjoint"]])
d_ctrl = np.array([x[0] for x in arms["control: same n, random drop"]])
for label, other in (("protein", b), ("both", c), ("control", d_ctrl)):
    d = other - a
    print(f"  {label:<8} vs scaffold: {d.mean():+.3f} +- {d.std(ddof=1):.3f}  "
          f"p = {stats.ttest_1samp(d,0).pvalue:.3f}")
dd = c - d_ctrl
print(f"  both disjoint vs its same-size control: {dd.mean():+.3f} +- {dd.std(ddof=1):.3f}  "
      f"p = {stats.ttest_1samp(dd,0).pvalue:.3f}   <- leakage net of data loss")
