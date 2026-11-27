"""The bar any pocket-to-ligand generator has to clear.

Given a pocket, the cheapest possible answer is: find the most similar pocket in
training and propose its ligand. No generation, no model. This project has twice
found a trivial baseline matching a sophisticated one, so the baseline is built
before the generator, not after.

Scored by ECFP4 Tanimoto between the proposed ligand and the true one, on a
protein-disjoint split -- on a random split the nearest training pocket is often
the same protein, and retrieval would look far better than it is.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caddack.qsar import parse_smiles
from caddack.qsar.descriptors import mol_to_ecfp_array

ROOT = Path("/home/user/pdbbind/v2015")
THRESH = 0.5


def pocket_residues(pid):
    f = ROOT / pid / f"{pid}_pocket.pdb"
    if not f.exists():
        return None
    return {(line[17:20].strip(), line[22:26].strip())
            for line in f.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 27} or None


def jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


blob = np.load("/home/user/pdbbind/affinity_features_refined_raw.npz", allow_pickle=True)
ids = list(blob["ids"])
smiles = list(blob["smiles"])
res = {p: r for p in ids if (r := pocket_residues(p)) is not None}
keep = [i for i, p in enumerate(ids) if p in res and parse_smiles(smiles[i]) is not None]
print(f"{len(keep)} complexes with both a pocket and a parseable ligand")

fp = {}
for i in keep:
    fp[i] = np.unpackbits(mol_to_ecfp_array(parse_smiles(smiles[i]), n_bits=1024))


def tanimoto(a, b):
    inter = np.bitwise_and(a, b).sum()
    return inter / (a.sum() + b.sum() - inter) if (a.sum() + b.sum() - inter) else 0.0


# Protein groups, so the split can hold whole proteins out.
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
            for b in members[i + 1:]:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                if jac(res[a], res[b]) >= THRESH:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
group = {i: find(ids[i]) for i in keep}

rows = []
for seed in range(3):
    rng = random.Random(seed)
    uniq = list({group[i] for i in keep})
    rng.shuffle(uniq)
    test_groups, n = set(), 0
    for g in uniq:
        if n >= 0.2 * len(keep):
            break
        test_groups.add(g)
        n += sum(1 for i in keep if group[i] == g)
    te = [i for i in keep if group[i] in test_groups]
    tr = [i for i in keep if group[i] not in test_groups]

    nearest, rand, best_possible = [], [], []
    for i in te:
        sims = [(jac(res[ids[i]], res[ids[j]]), j) for j in tr]
        _, j = max(sims)
        nearest.append(tanimoto(fp[i], fp[j]))
        rand.append(tanimoto(fp[i], fp[rng.choice(tr)]))
        best_possible.append(max(tanimoto(fp[i], fp[j]) for j in tr))
    rows.append((np.mean(nearest), np.mean(rand), np.mean(best_possible),
                 np.mean([x > 0.4 for x in nearest]), len(te)))
    print(f"  seed {seed}: test {len(te)}  nearest-pocket {rows[-1][0]:.3f}  "
          f"random {rows[-1][1]:.3f}  best-in-train {rows[-1][2]:.3f}", flush=True)

a = np.array(rows)
print("\nECFP4 Tanimoto of the proposed ligand to the true one, protein-disjoint:")
print(f"  retrieve nearest pocket's ligand   {a[:,0].mean():.3f} +- {a[:,0].std(ddof=1):.3f}")
print(f"  random training ligand             {a[:,1].mean():.3f} +- {a[:,1].std(ddof=1):.3f}")
print(f"  best ligand anywhere in training   {a[:,2].mean():.3f} +- {a[:,2].std(ddof=1):.3f}"
      "   <- ceiling for any retrieval method")
print(f"  retrievals above Tanimoto 0.4      {a[:,3].mean()*100:.0f}%")
