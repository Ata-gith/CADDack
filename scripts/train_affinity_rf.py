#!/usr/bin/env python3
"""Train the affinity model that ships with CADDack, and export it for the demo.

A random forest on 115 interaction features. This replaced a two-tower Bayesian
graph network, which is still in caddack.gnn: on a PDBbind scaffold split the
forest reaches r 0.696 and MAE 1.067 against the network's 0.644 and 1.268, and
stayed ahead when the network was handed the same features.

    # one-time, builds the cached complexes
    python scripts/benchmark_pdbbind.py --data-dir /path/to/pdbbind --download --build

    python scripts/train_affinity_rf.py --data-dir /path/to/pdbbind

Writes models/affinity/ for Python use and docs/affinity.json for the browser.
Metrics are reported over several scaffold splits with a standard deviation,
because single-split differences on this data are routinely smaller than the
spread between splits.
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caddack.affinity.features import (  # noqa: E402
    feature_names,
    features_from_atoms,
    features_from_files,
)
from caddack.qsar import canonicalize_smiles  # noqa: E402
from caddack.qsar.split import murcko_scaffold, scaffold_split  # noqa: E402

CACHE_NAME = "pdbbind_examples.pkl"


def build_matrix(examples, cache: Path | None, data_dir: Path, from_raw: bool,
                 fp_bits: int = 0):
    """Featurise every complex once, caching the result.

    Featurisation is the slow part and depends only on the structures, never on
    the split or the model, so a rebuild with different hyperparameters should
    not pay for it again.
    """
    if cache is not None and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        if "mask" not in blob:          # written before the split fix
            cache.unlink()
        elif int(blob.get("fp_bits", 0)) != fp_bits:
            # A cache built with a different fingerprint width describes a
            # different feature space. Silently reusing it would train the model
            # on one set of columns and score the browser on another.
            print(f"  features: cache has fp_bits={int(blob.get('fp_bits', 0))}, "
                  f"need {fp_bits}; rebuilding")
            cache.unlink()
        else:
            print(f"  features: loaded {blob['X'].shape} from {cache.name}")
            return (blob["X"], blob["y"], list(blob["ids"]),
                    list(blob["smiles"]), blob["mask"])

    # A validity mask over the FULL example list, not a filtered copy. The split
    # has to be computed on the same list the graph model splits, or the two are
    # scored on different molecules and cannot be compared -- which is exactly
    # what happened: their test sets overlapped on only 458 of 740.
    rows, targets, ids, smiles, valid = [], [], [], [], []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        if from_raw:
            # The cached geometry keeps only ATOM records within 6 A, dropping
            # cofactors and metals and clipping the longer contact shells. Reading
            # the pocket files again is slower but measurably better: r 0.696
            # against 0.685 on the same split.
            base = data_dir / "v2015" / ex.pdb_id
            x = features_from_files(base / f"{ex.pdb_id}_pocket.pdb",
                                    base / f"{ex.pdb_id}_ligand.mol2",
                                    fp_bits=fp_bits)
        else:
            x = features_from_atoms(ex.geo.atomic_nums, ex.geo.positions,
                                    n_ligand=ex.geo.n_ligand, smiles=ex.ligand_smiles,
                                    fp_bits=fp_bits)
        valid.append(x is not None)
        if x is None:
            continue
        rows.append(x)
        targets.append(ex.affinity)
        ids.append(ex.pdb_id)
        smiles.append(ex.ligand_smiles)
        if (i + 1) % 1000 == 0:
            print(f"    ...{i + 1}/{len(examples)} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    X = np.asarray(rows)
    y = np.asarray(targets)
    mask = np.asarray(valid, dtype=bool)
    print(f"  features: {X.shape} in {time.perf_counter() - t0:.0f}s"
          f"{'' if mask.all() else f' ({(~mask).sum()} complexes unreadable)'}")
    if cache is not None:
        np.savez_compressed(cache, X=X, y=y, ids=np.array(ids),
                            smiles=np.array(smiles, dtype=object), mask=mask,
                            fp_bits=fp_bits)
    return X, y, ids, smiles, mask


def _scaffold_key(smi: str):
    """Murcko scaffold, or the canonical SMILES when the ligand has no rings.

    murcko_scaffold returns None for acyclic molecules. Lumping those under a
    single key would drop every acyclic augmentation complex the moment one
    acyclic ligand landed in the test set.
    """
    if not smi:
        return None
    sc = murcko_scaffold(smi)
    return sc if sc else (canonicalize_smiles(smi) or smi)


def augmentation_rows(extra_pool, test_rows, smiles):
    """Rows from ``extra_pool`` that are safe to add to training for this split.

    Any complex whose ligand shares a Bemis-Murcko scaffold with a test ligand is
    dropped. Without that filter the extra data leaks: the general set contains
    close analogues of refined complexes, so adding it unfiltered scores the
    model on chemistry it has effectively already seen.
    """
    test_keys = {_scaffold_key(smiles[r]) for r in test_rows} - {None}
    keep = [r for r in extra_pool
            if (k := _scaffold_key(smiles[r])) is not None and k not in test_keys]
    return np.array(keep, dtype=int), len(extra_pool) - len(keep)


def evaluate(X, y, smiles, all_smiles, mask, args, extra_pool=None) -> tuple[dict, dict]:
    """Repeated scaffold splits. Returns summary stats and the base fold's detail.

    The split runs over every complex in the dataset, then is restricted to the
    ones that featurised. Splitting a pre-filtered list instead would put these
    models on a different test set from the graph models, which are split the
    same way.

    ``extra_pool`` holds rows that may join *training* only -- the general-set
    complexes under --augment-general. The test set stays whatever the split
    produced from ``all_smiles``, so the numbers stay comparable with every other
    model measured here.
    """
    position = np.cumsum(mask) - 1          # dataset index -> row in X
    per_fold, detail = [], {}
    for rep in range(args.n_repeats):
        seed = args.seed + rep
        frame = pd.DataFrame({"smiles": all_smiles})
        tr_all, te_all = scaffold_split(frame, smiles_col="smiles",
                                        test_size=args.test_size, seed=seed)
        tr = position[[i for i in tr_all if mask[i]]]
        te = position[[i for i in te_all if mask[i]]]
        if extra_pool is not None and len(extra_pool):
            extra, dropped = augmentation_rows(extra_pool, te, smiles)
            print(f"    split {seed}: +{len(extra)} general complexes for training "
                  f"({dropped} dropped for scaffold overlap with test)")
            tr = np.concatenate([tr, extra])
        rf = RandomForestRegressor(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   min_samples_leaf=args.min_samples_leaf,
                                   random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        fold = {
            "pearson": float(pearsonr(pred, y[te]).statistic),
            "r2": float(1 - np.sum((pred - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)),
            "mae": float(np.mean(np.abs(pred - y[te]))),
            "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
            "baseline_mae": float(np.mean(np.abs(y[tr].mean() - y[te]))),
            "n_train": int(len(tr)), "n_test": int(len(te)), "seed": seed,
        }
        per_fold.append(fold)
        print(f"    split {seed}: r {fold['pearson']:+.3f}  R2 {fold['r2']:+.3f}  "
              f"MAE {fold['mae']:.3f}")
        if rep == 0:
            # The exported fold is the base seed's, so the page shows the same
            # held-out molecules the graph model's bundle does. Keeping the last
            # repeat instead put the two bundles on different splits, overlapping
            # on 458 of 740 complexes, and made the comparison meaningless.
            detail = {"train_idx": tr, "test_idx": te, "pred": pred, "model": rf}

    summary = {k: {"mean": float(np.mean([f[k] for f in per_fold])),
                   "sd": float(np.std([f[k] for f in per_fold], ddof=1))
                   if len(per_fold) > 1 else 0.0}
               for k in ("pearson", "r2", "mae", "rmse", "baseline_mae")}
    summary["n_repeats"] = args.n_repeats
    return summary, detail


def export_forest(rf) -> dict:
    """Flatten the trees into typed arrays so a browser can walk them."""
    def b64(a):
        return base64.b64encode(a.tobytes()).decode()

    feature, threshold, left, right, value, offsets = [], [], [], [], [], [0]
    for est in rf.estimators_:
        t = est.tree_
        feature.append(t.feature.astype(np.int16))
        threshold.append(t.threshold.astype(np.float32))
        left.append(t.children_left.astype(np.int32))
        right.append(t.children_right.astype(np.int32))
        value.append(t.value.reshape(-1).astype(np.float32))
        offsets.append(offsets[-1] + t.node_count)
    return {"n_trees": len(rf.estimators_), "offsets": offsets,
            "feature": b64(np.concatenate(feature)),
            "threshold": b64(np.concatenate(threshold)),
            "left": b64(np.concatenate(left)),
            "right": b64(np.concatenate(right)),
            "value": b64(np.concatenate(value))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "models" / "affinity")
    ap.add_argument("--demo-out", type=Path, default=REPO_ROOT / "docs" / "affinity.json")
    # The Python model is unrestricted: nobody downloads it before using it.
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    # The browser copy is capped, because it is fetched before anything can be
    # scored. 100 trees at depth 18 cost about 0.01 in r and shrink the bundle
    # from 20 MB to under 2.
    ap.add_argument("--demo-n-estimators", type=int, default=100)
    ap.add_argument("--demo-max-depth", type=int, default=18)
    ap.add_argument("--demo-min-samples-leaf", type=int, default=4)
    ap.add_argument("--from-cache-geometry", dest="from_raw", action="store_false",
                    default=True,
                    help="featurise from the cached 6 A pockets instead of the raw "
                         "pocket files; faster, and measurably worse")
    ap.add_argument("--fingerprint-bits", type=int, default=1024,
                    help="ECFP4 bits appended to the structural features (0 "
                         "disables). The contact counts describe the interface "
                         "but discard which ligand made it; the bits restore "
                         "that. 1024 lifts r 0.686 -> 0.704 over five splits.")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--general", action="store_true",
                    help="train AND test on the full general set rather than the "
                         "refined subset")
    ap.add_argument("--augment-general", action="store_true",
                    help="keep the refined scaffold split for the test set, but add "
                         "the general-set complexes to TRAINING, after removing any "
                         "that share a scaffold with the test set. Takes training "
                         "from 2,959 to about 10,800: aggregate accuracy barely "
                         "moves, but test ligands with no close training analogue "
                         "fall from 184 to 85 of 740.")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    with open(args.data_dir / CACHE_NAME, "rb") as fh:
        examples = pickle.load(fh)
    if args.general and args.augment_general:
        raise SystemExit("--general and --augment-general are mutually exclusive: "
                         "the first moves the test set, the second does not.")
    if not (args.general or args.augment_general):
        examples = [e for e in examples if getattr(e, "is_refined", True)]

    n_refined = len(examples)
    if args.augment_general:
        # Refined complexes first, so a scaffold split over the refined prefix
        # reproduces the refined-only split exactly and the test set is unchanged.
        # The cache is tagged separately because it is keyed by filename alone:
        # reusing the general cache, which was written in the original order,
        # would silently pair every row with the wrong complex.
        refined = [e for e in examples if getattr(e, "is_refined", True)]
        general = [e for e in examples if not getattr(e, "is_refined", True)]
        examples = refined + general
        n_refined = len(refined)
        print(f"  {len(examples)} complexes: {n_refined} refined (split as usual) "
              f"+ {len(general)} general (training only)")
    else:
        print(f"  {len(examples)} complexes"
              f"{'' if args.general else ' (refined set)'}")

    tag = ("general_aug" if args.augment_general
           else "general" if args.general else "refined")
    tag += "_raw" if args.from_raw else ""
    cache = None if args.no_cache else args.data_dir / f"affinity_features_{tag}.npz"
    X, y, ids, smiles, mask = build_matrix(examples, cache, args.data_dir,
                                           args.from_raw, args.fingerprint_bits)
    all_smiles = [e.ligand_smiles or "" for e in examples]
    extra_pool = None
    if args.augment_general:
        # Split over the refined prefix only; the general tail joins training.
        # Rows are in example order, so the refined rows are the first
        # mask[:n_refined].sum() of them.
        n_valid_refined = int(mask[:n_refined].sum())
        extra_pool = np.arange(n_valid_refined, int(mask.sum()))
        all_smiles = all_smiles[:n_refined]
        mask = mask[:n_refined]
        assert len(all_smiles) == len(mask) == n_refined

    print(f"  evaluating over {args.n_repeats} scaffold splits")
    summary, detail = evaluate(X, y, smiles, all_smiles, mask, args, extra_pool)
    print("  summary: " + "  ".join(
        f"{k} {v['mean']:+.3f}±{v['sd']:.3f}" for k, v in summary.items()
        if isinstance(v, dict)))

    # The shipped model is fitted on everything; the metrics above describe how
    # such a model behaves on chemistry it has not seen.
    final = RandomForestRegressor(n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  min_samples_leaf=args.min_samples_leaf,
                                  random_state=args.seed, n_jobs=-1)
    final.fit(X, y)

    args.outdir.mkdir(parents=True, exist_ok=True)
    dump(final, args.outdir / "forest.joblib")
    (args.outdir / "metadata.json").write_text(json.dumps({
        "feature_names": feature_names(args.fingerprint_bits),
        "fingerprint_bits": args.fingerprint_bits,
        "metrics": summary,
        "n_training_complexes": int(len(y)),
        "dataset": ("PDBbind v2015 refined, trained with the general set"
                    if args.augment_general else
                    "PDBbind v2015 " + ("general" if args.general else "refined")),
        "target": "-log Kd/Ki (pK units)",
    }, indent=2), encoding="utf-8")
    print(f"  wrote {args.outdir}/")

    # Browser bundle: a capped forest, fitted on the same training split so the
    # predictions shown are genuinely held out.
    tr, te = detail["train_idx"], detail["test_idx"]
    demo = RandomForestRegressor(n_estimators=args.demo_n_estimators,
                                 max_depth=args.demo_max_depth,
                                 min_samples_leaf=args.demo_min_samples_leaf,
                                 random_state=args.seed, n_jobs=-1)
    demo.fit(X[tr], y[tr])
    demo_per_tree = np.stack([t.predict(X[te]) for t in demo.estimators_])
    demo_pred, demo_spread = demo_per_tree.mean(0), demo_per_tree.std(0, ddof=0)
    demo_metrics = {
        "pearson": float(pearsonr(demo_pred, y[te]).statistic),
        "r2": float(1 - np.sum((demo_pred - y[te]) ** 2)
                    / np.sum((y[te] - y[te].mean()) ** 2)),
        "mae": float(np.mean(np.abs(demo_pred - y[te]))),
        "n_trees": args.demo_n_estimators, "max_depth": args.demo_max_depth,
        "min_samples_leaf": args.demo_min_samples_leaf,
        "coverage_1sd": float(np.mean(np.abs(demo_pred - y[te]) <= demo_spread)),
        "coverage_2sd": float(np.mean(np.abs(demo_pred - y[te]) <= 2 * demo_spread)),
    }

    # The per-complex predictions shipped to the page come from the FULL forest,
    # the one whose metrics the page quotes -- not from the capped browser copy.
    # Depth caps make the browser copy's trees agree with one another far more
    # than the full forest's do, so its spread is too narrow to read as a
    # standard deviation: 53% of measurements inside +-1 at depth 14 against 74%
    # unconstrained. Shipping the capped copy's spread would have put a number on
    # the uncertainty panel that describes neither the quoted model nor a
    # calibrated interval.
    full = detail["model"]
    per_tree = np.stack([t.predict(X[te]) for t in full.estimators_])
    pred, spread = per_tree.mean(0), per_tree.std(0, ddof=0)
    print(f"  browser copy ({args.demo_n_estimators} trees, depth "
          f"{args.demo_max_depth}): r {demo_metrics['pearson']:+.3f}  "
          f"MAE {demo_metrics['mae']:.3f}")
    bundle = {
        "forest": export_forest(demo),
        "feature_names": feature_names(args.fingerprint_bits),
        "fingerprint_bits": args.fingerprint_bits,
        "metrics": summary,
        "demo_metrics": demo_metrics,
        "complex_source": ("full forest, "
                           f"{args.n_estimators} trees, seed {args.seed}"),
        "dataset": ("PDBbind v2015 refined, trained with the general set"
                    if args.augment_general else
                    "PDBbind v2015 " + ("general" if args.general else "refined")),
        "n_train": int(len(tr)),
        "complexes": [{"id": ids[i], "measured": round(float(y[i]), 3),
                       "pred": round(float(p), 3), "spread": round(float(s), 3)}
                      for i, p, s in zip(te, pred, spread)],
    }
    args.demo_out.parent.mkdir(parents=True, exist_ok=True)
    args.demo_out.write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    print(f"  wrote {args.demo_out} ({args.demo_out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
