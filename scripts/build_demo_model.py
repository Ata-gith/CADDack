#!/usr/bin/env python3
"""Build the model that powers the browser demo in ``docs/``.

Trains a random forest to predict aqueous solubility (log S) on the ESOL
dataset, using CADDack's own featurisation, scaffold splitting and
applicability-domain code, then exports everything the static page needs into
``docs/model.json``.

    python scripts/build_demo_model.py            # downloads ESOL, writes docs/model.json
    python scripts/build_demo_model.py --csv path/to/delaney-processed.csv

Why a random forest exported tree-by-tree, rather than ONNX: the demo's whole
point is showing an honest uncertainty estimate, and the spread across trees
gives one directly. ONNX Runtime Web would return only the ensemble mean, and
cost ~10 MB of runtime to do it. Walking the trees in JavaScript is ~15 lines.

The browser must reproduce these features exactly. Two details are load-bearing
and are covered by ``tests/test_demo_parity.py``:

  * Fingerprints are Morgan radius 2 with chirality. RDKit.js spells that flag
    ``useChirality``; it silently ignores ``includeChirality`` (the Python
    spelling) and returns different bits.
  * The feature vector is ``[*NUM_DESC, ECFP bits...]`` in exactly this order.

The page also needs RDKit compiled to WebAssembly, which is vendored separately
and is not produced by this script::

    npm pack @rdkit/rdkit
    tar xzf rdkit-rdkit-*.tgz
    cp package/dist/RDKit_minimal.{js,wasm} docs/
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caddack.qsar.applicability import SimilarityAD  # noqa: E402
from caddack.qsar.descriptors import (  # noqa: E402
    featurize_dataframe,
    mol_to_ecfp_array,
    parse_smiles,
)
from caddack.qsar.split import scaffold_split  # noqa: E402
from caddack.qsar.train_qsar import NUM_DESC, _metrics_reg  # noqa: E402

ESOL_URL = ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
            "delaney-processed.csv")
TARGET_COL = "measured log solubility in mols per litre"


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode()


def load_esol(csv: Path | None, cache_dir: Path) -> pd.DataFrame:
    if csv is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        csv = cache_dir / "delaney-processed.csv"
        if not csv.exists():
            print(f"  downloading ESOL -> {csv}")
            urllib.request.urlretrieve(ESOL_URL, csv)
    df = pd.read_csv(csv).rename(columns={TARGET_COL: "logS", "smiles": "SMILES"})
    return df[["SMILES", "logS"]]


def export_forest(rf: RandomForestRegressor) -> dict:
    """Flatten every tree into one set of typed arrays, base64-encoded.

    Trees are stored back to back with an offset table rather than nested, so
    the browser decodes five flat arrays instead of walking a JSON object graph.
    """
    feature, threshold, left, right, value, offsets = [], [], [], [], [], [0]
    for est in rf.estimators_:
        t = est.tree_
        feature.append(t.feature.astype(np.int16))
        threshold.append(t.threshold.astype(np.float32))
        left.append(t.children_left.astype(np.int32))
        right.append(t.children_right.astype(np.int32))
        value.append(t.value.reshape(-1).astype(np.float32))
        offsets.append(offsets[-1] + t.node_count)
    return {
        "n_trees": len(rf.estimators_),
        "offsets": offsets,
        "feature": _b64(np.concatenate(feature)),
        "threshold": _b64(np.concatenate(threshold)),
        "left": _b64(np.concatenate(left)),
        "right": _b64(np.concatenate(right)),
        "value": _b64(np.concatenate(value)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="ESOL CSV; downloaded to --cache-dir when omitted")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "molnet")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "model.json")
    ap.add_argument("--n-bits", type=int, default=1024,
                    help="fingerprint length; 1024 keeps the bundle small")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = load_esol(args.csv, args.cache_dir)
    print(f"  ESOL: {len(df)} molecules")

    feat = featurize_dataframe(df, smiles_col="SMILES", target_col="logS",
                               n_bits=args.n_bits)
    if "__error" in feat.columns:
        feat = feat[feat["__error"].isna()].reset_index(drop=True)
    print(f"  featurised: {len(feat)} valid")

    bits = [f"ECFP4_{i}" for i in range(args.n_bits)]
    X = feat[NUM_DESC + bits].to_numpy(np.float32)
    y = feat["logS"].to_numpy(np.float32)

    # Scaffold split, not random: test molecules must be structurally novel or
    # the reported R2 is just measuring memorisation of a congeneric series.
    train_idx, test_idx, report = scaffold_split(
        feat, smiles_col="SMILES_canonical", test_size=args.test_size,
        seed=args.seed, return_report=True)
    print(f"  scaffold split: train={len(train_idx)} test={len(test_idx)} "
          f"achieved={report.achieved_test_size:.3f} "
          f"groups={report.n_scaffold_groups}")

    rf = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, random_state=args.seed, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])

    metrics = _metrics_reg(y[test_idx], rf.predict(X[test_idx]))
    print("  held-out (scaffold):", {k: round(v, 3) for k, v in metrics.items()
                                     if isinstance(v, float)})

    train_smiles = feat.loc[train_idx, "SMILES_canonical"].tolist()
    packed = np.stack([mol_to_ecfp_array(parse_smiles(s), n_bits=args.n_bits)
                       for s in train_smiles])
    ad = SimilarityAD.fit(np.unpackbits(packed, axis=1), percentile=5.0)
    print(f"  applicability domain: threshold={ad.threshold:.3f}")

    bundle = {
        "forest": export_forest(rf),
        "train": {
            "n": len(train_smiles),
            "n_bits": args.n_bits,
            # Packed to one bit per bit: 128 bytes a molecule instead of 8 KB.
            "fps": _b64(packed.astype(np.uint8)),
            "logS": _b64(y[train_idx].astype(np.float32)),
            "smiles": train_smiles,
        },
        "ad": {"threshold": float(ad.threshold), "percentile": float(ad.percentile)},
        "feature_order": list(NUM_DESC),
        "n_bits": args.n_bits,
        "metrics": metrics,
        "split": report.as_dict(),
        "dataset": {"name": "ESOL (Delaney)", "target": "log S (mol/L)",
                    "source": ESOL_URL},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"  wrote {args.out} ({size:.2f} MB raw; GitHub Pages serves it gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
