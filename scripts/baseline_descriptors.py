#!/usr/bin/env python3
"""The bar any structure-aware affinity model has to clear.

A random forest on nine scalars -- six standard descriptors plus the ligand's
heavy-atom count, the pocket's atom count, and their ratio -- uses no 3D
information beyond how many protein atoms sit within the cutoff. On the PDBbind
refined set with a scaffold split it reaches r = 0.585, R2 = 0.339, MAE = 1.208.

That matched the two-tower Bayesian network, which is how the network was found
to be ignoring the structure: sum-pooling an undifferentiated bag of ligand and
pocket atoms is a learned atom count, so an atom inventory minimised the loss
and the geometry never had to be used.

Run this alongside any change to the affinity model. A model that does not clear
these numbers is not using the structure, whatever its architecture diagram says.
Compare over several seeds: run-to-run spread on this split is about 0.08 in R2,
which is larger than most architectural effects.

    python scripts/baseline_descriptors.py --data-dir /path/to/pdbbind
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caddack.qsar.descriptors import mol_to_basic_features, smiles_to_mol_clean  # noqa: E402
from caddack.qsar.split import scaffold_split  # noqa: E402

CACHE_NAME = "pdbbind_examples.pkl"
FEATURES = ["MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds",
            "n_ligand_atoms", "n_pocket_atoms", "pocket_per_ligand_atom"]


def featurize(examples, indices):
    rows, targets = [], []
    for i in indices:
        ex = examples[i]
        mol = smiles_to_mol_clean(ex.ligand_smiles) if ex.ligand_smiles else None
        if mol is None:
            continue
        desc = mol_to_basic_features(mol)
        n_lig = mol.GetNumAtoms()
        rows.append([desc["MolWt"], desc["LogP"], desc["TPSA"], desc["NumHBD"],
                     desc["NumHBA"], desc["NumRotBonds"], n_lig,
                     ex.geo.n_pocket, ex.geo.n_pocket / max(n_lig, 1)])
        targets.append(ex.affinity)
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def score(pred, truth) -> dict:
    return {
        "pearson": float(pearsonr(pred, truth).statistic),
        "r2": float(1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="write metrics as JSON")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="split seed; match the model's")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--general", action="store_true",
                    help="use the full general set instead of the refined subset")
    args = ap.parse_args(argv)

    with open(args.data_dir / CACHE_NAME, "rb") as fh:
        examples = pickle.load(fh)
    if not args.general:
        examples = [e for e in examples if getattr(e, "is_refined", True)]

    frame = pd.DataFrame({"smiles": [e.ligand_smiles for e in examples]})
    train_idx, test_idx = scaffold_split(frame, smiles_col="smiles",
                                         test_size=args.test_size, seed=args.seed)
    X_train, y_train = featurize(examples, train_idx)
    X_test, y_test = featurize(examples, test_idx)
    print(f"  {len(examples)} complexes, scaffold split: "
          f"train {len(y_train)}  test {len(y_test)}")
    print(f"  features: {', '.join(FEATURES)}\n")

    results = {}
    models = [
        ("ridge regression", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ("random forest", RandomForestRegressor(n_estimators=args.n_estimators,
                                                random_state=args.seed, n_jobs=-1)),
    ]
    for name, model in models:
        model.fit(X_train, y_train)
        results[name] = score(model.predict(X_test), y_test)
        m = results[name]
        print(f"    {name:<20} r {m['pearson']:+.3f}   R2 {m['r2']:+.3f}   "
              f"MAE {m['mae']:.3f}   RMSE {m['rmse']:.3f}")

    mean_pred = np.full_like(y_test, y_train.mean())
    results["predict the mean"] = score(mean_pred, y_test)
    print(f"    {'predict the mean':<20} r {'   n/a':>6}   "
          f"R2 {results['predict the mean']['r2']:+.3f}   "
          f"MAE {results['predict the mean']['mae']:.3f}")

    rf = results["random forest"]
    print(f"\n  A structure-aware model should beat r {rf['pearson']:.3f} / "
          f"MAE {rf['mae']:.3f}, over several seeds.")

    if args.out:
        args.out.write_text(json.dumps(
            {"features": FEATURES, "n_train": len(y_train), "n_test": len(y_test),
             "results": results}, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
