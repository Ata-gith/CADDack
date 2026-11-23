#!/usr/bin/env python3
"""The bar any structure-aware affinity model has to clear.

A random forest on nine scalars -- six standard descriptors plus the ligand's
heavy-atom count, the pocket's atom count, and their ratio -- uses no 3D
information beyond how many protein atoms sit within the cutoff. On the PDBbind
refined set with a scaffold split it reaches r = 0.585, R2 = 0.339, MAE = 1.208.

--vina-terms adds Vina's five interaction terms, which do use the geometry:
r = 0.636, R2 = 0.403, MAE = 1.154.

--contact-features adds 100 more -- ligand-protein pair counts split by element
class and distance shell -- and reaches r = 0.696, R2 = 0.478, MAE = 1.067,
averaged over three split seeds and improving on all three. That is the bar now,
and it is a substantial one: it beats the two-tower Bayesian network, which
manages r 0.644 and MAE 1.268 on the same split.

The pattern across all of it is that describing atom *pairs* beats describing
atoms. Five Vina terms helped; a hundred binned contact counts helped more. A
network given only atoms has to infer pairwise structure from 2,959 examples,
and on this evidence it does not manage.

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

from caddack.docking.scoring import (  # noqa: E402
    TERM_NAMES,
    contact_features,
    vina_terms,
)
from caddack.gnn.geometry import load_ligand, parse_pdb_atoms  # noqa: E402
from caddack.qsar.descriptors import mol_to_basic_features, smiles_to_mol_clean  # noqa: E402
from caddack.qsar.split import scaffold_split  # noqa: E402

CACHE_NAME = "pdbbind_examples.pkl"
FEATURES = ["MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds",
            "n_ligand_atoms", "n_pocket_atoms", "pocket_per_ligand_atom"]


def structure_features(data_dir: Path, pdb_id: str, cutoff: float = 8.0,
                       with_contacts: bool = False):
    """Vina's interaction terms for one complex, from the raw pocket file.

    Recomputed from disk rather than taken from the cached GeometryRecord: the
    cache keeps only ATOM records within 6 A, which drops cofactors and metals
    and truncates the longer-range gauss2 term. Doing it properly is worth about
    0.2 in correlation with affinity.
    """
    base = data_dir / "v2015" / pdb_id
    protein = parse_pdb_atoms(base / f"{pdb_id}_pocket.pdb")
    ligand = load_ligand(base / f"{pdb_id}_ligand.mol2")
    if not protein or not ligand:
        return None
    lig_pos = np.array([[a.x, a.y, a.z] for a in ligand])
    lig_z = [a.atomic_num for a in ligand]
    pro_pos = np.array([[a.x, a.y, a.z] for a in protein])
    pro_z = np.array([a.atomic_num for a in protein])
    near = np.sqrt(((pro_pos[:, None, :] - lig_pos[None, :, :]) ** 2).sum(-1)).min(1) <= cutoff
    z = list(lig_z) + list(pro_z[near])
    xyz = np.vstack([lig_pos, pro_pos[near]])
    terms = vina_terms(z, xyz, n_ligand=len(lig_z))
    out = [terms[k] for k in TERM_NAMES] + [terms["vina_inter"]]
    if with_contacts:
        out += contact_features(z, xyz, n_ligand=len(lig_z)).tolist()
    return out


def featurize(examples, indices, data_dir=None, with_vina=False,
              with_contacts=False):
    rows, targets = [], []
    for i in indices:
        ex = examples[i]
        mol = smiles_to_mol_clean(ex.ligand_smiles) if ex.ligand_smiles else None
        if mol is None:
            continue
        extra = []
        if with_vina:
            extra = structure_features(data_dir, ex.pdb_id, with_contacts=with_contacts)
            if extra is None:
                continue
        desc = mol_to_basic_features(mol)
        n_lig = mol.GetNumAtoms()
        rows.append([desc["MolWt"], desc["LogP"], desc["TPSA"], desc["NumHBD"],
                     desc["NumHBA"], desc["NumRotBonds"], n_lig,
                     ex.geo.n_pocket, ex.geo.n_pocket / max(n_lig, 1)] + extra)
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
    ap.add_argument("--vina-terms", action="store_true",
                    help="add Vina's five interaction terms and their weighted "
                         "sum, recomputed from the raw pocket files")
    ap.add_argument("--contact-features", action="store_true",
                    help="also add distance-binned contact counts by element-class "
                         "pair (implies --vina-terms)")
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
    if args.contact_features:
        args.vina_terms = True
    kw = {"data_dir": args.data_dir, "with_vina": args.vina_terms,
          "with_contacts": args.contact_features}
    X_train, y_train = featurize(examples, train_idx, **kw)
    X_test, y_test = featurize(examples, test_idx, **kw)
    print(f"  {len(examples)} complexes, scaffold split: "
          f"train {len(y_train)}  test {len(y_test)}")
    names = FEATURES + ([*TERM_NAMES, "vina_inter"] if args.vina_terms else [])
    if args.contact_features:
        from caddack.docking.scoring import contact_feature_names
        names = names + contact_feature_names()
    print(f"  features: {len(names)}"
          f"{'' if len(names) > 30 else ' -- ' + ', '.join(names)}\n")

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
    kind = "descriptors + Vina terms" if args.vina_terms else "descriptors alone"
    print(f"\n  Bar from {kind}: a structure-aware model should beat "
          f"r {rf['pearson']:.3f} / MAE {rf['mae']:.3f}, over several seeds.")

    if args.out:
        args.out.write_text(json.dumps(
            {"features": FEATURES, "n_train": len(y_train), "n_test": len(y_test),
             "results": results}, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
