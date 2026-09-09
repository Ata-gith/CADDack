#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from caddack.qsar.split import scaffold_split

NUM_DESC = ["MolWt","LogP","TPSA","NumHBD","NumHBA","NumRotBonds"]


def add_cli(subparsers):
    p = subparsers.add_parser("qsar-train", help="Train baseline QSAR on Parquet features")
    p.add_argument("--parquet", required=True, help="Input features parquet")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", choices=["regression", "classification"], default=None,
                   help="If omitted, auto-detect from the target column")
    p.add_argument("--split", choices=["random","scaffold"], default="scaffold")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="models/qsar", help="Output directory")
    p.add_argument("--max-features", type=str, default="auto", help="RandomForest max_features")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-repeats", type=int, default=5,
                   help="repeated splits with consecutive seeds; metrics are "
                        "reported as mean +/- sd so a number carries its own "
                        "uncertainty (a single split on a few hundred compounds "
                        "has sampling noise comparable to the effect measured)")
    p.set_defaults(func=run)


def _select_features(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Descriptor + fingerprint columns, with the target excluded.

    Without the exclusion, training on a descriptor target (MolWt, LogP, ...)
    puts the answer in the feature matrix -- textbook target leakage.
    """
    bits = [c for c in df.columns if c.startswith("ECFP")]
    cols = [c for c in NUM_DESC if c in df.columns and c != target] + bits
    if not cols:
        raise SystemExit(f"no usable features after excluding target {target!r}")
    return df[cols]


def _auto_task(y: pd.Series) -> str:
    if y.dropna().nunique() <= 2:
        return "classification"
    return "regression"


def _drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid_smiles rows and drop the __error column."""
    if "__error" in df.columns:
        df = df[df["__error"] != "invalid_smiles"].copy()
        df = df.drop(columns=["__error"])
    return df


def _metrics_reg(y_true, y_pred) -> dict:
    """Regression metrics.

    R2 and Pearson r answer different questions: R2 penalises bias and scale
    error, r measures linear association only. A model can rank well (high r)
    while being badly calibrated (R2 near zero), so both are reported.
    """
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_test": int(len(y_true)),
    }
    # Pearson r is undefined when either vector is constant.
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, p = pearsonr(y_true, y_pred)
        out["pearson_r"], out["pearson_p"] = float(r), float(p)
    else:
        out["pearson_r"], out["pearson_p"] = None, None
    return out


def _parse_max_features(value: str):
    """Interpret --max-features: keep sklearn keywords, coerce numerics."""
    if value in ("auto", "sqrt", "log2"):
        return "sqrt" if value == "auto" else value
    try:
        f = float(value)
        return int(f) if f.is_integer() and f > 1 else f
    except ValueError:
        return value  # let sklearn raise a clear error on a bad keyword


def _metrics_clf(y_true, y_prob, y_pred) -> dict:
    out = {
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }
    # roc_auc is undefined when the test set has a single class
    if len(np.unique(y_true)) >= 2:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auc_roc"] = None
    return out


MIN_TEST_ROWS = 5


def _json_safe(obj):
    """json.dumps default= hook: NaN/Inf are not valid JSON (RFC 8259)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return v if not (isinstance(v, float) and not math.isfinite(v)) else None
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON that other languages can actually parse.

    Python emits bare NaN/Infinity by default and reads them back happily, so the
    problem is invisible here and only shows up in JavaScript, Go or Rust.
    """
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_safe),
        encoding="utf-8",
    )


def _split_once(df_masked, X, y, task, args, seed):
    """One train/test split. Returns (X_train, X_test, y_train, y_test, report)."""
    if args.split == "scaffold":
        if "SMILES_canonical" not in df_masked.columns:
            raise SystemExit("SMILES_canonical required for scaffold split")
        train_idx, test_idx, report = scaffold_split(
            df_masked, smiles_col="SMILES_canonical",
            test_size=args.test_size, seed=seed, return_report=True,
        )
        return (X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx], report.as_dict())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=seed,
        stratify=(y if task == "classification" else None),
    )
    return X_tr, X_te, y_tr, y_te, None


def _fit_eval(X_train, X_test, y_train, y_test, task, args, seed):
    """Fit one model and score it. Returns (model, metrics)."""
    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators, random_state=seed, n_jobs=-1,
            max_features=_parse_max_features(args.max_features),
        )
        model.fit(X_train, y_train)
        return model, _metrics_reg(y_test, model.predict(X_test))

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    uniq = np.unique(y_train)
    if uniq.size < 2:
        raise SystemExit(
            f"classification task but only one class present in training data: {uniq.tolist()}"
        )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=seed, n_jobs=-1,
        class_weight="balanced",
        max_features=_parse_max_features(args.max_features),
    )
    model.fit(X_train, y_train)
    classes = model.classes_
    if classes.shape[0] != 2:
        raise SystemExit(f"expected binary classification, got classes={classes.tolist()}")
    pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
    y_prob = model.predict_proba(X_test)[:, pos_idx]
    return model, _metrics_clf(y_test, y_prob, (y_prob >= 0.5).astype(int))


def _summarise(folds: list[dict]) -> dict:
    """Mean and sd across repeats, so a number carries its own uncertainty."""
    keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float))
            and folds[0][k] is not None]
    out = {}
    for k in keys:
        vals = [f[k] for f in folds if isinstance(f.get(k), (int, float))]
        if not vals:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_repeats": len(vals),
        }
    return out


def run(args):
    inp = Path(args.parquet)
    df0 = pd.read_parquet(inp)
    df = _drop_invalid(df0)

    if args.target not in df.columns:
        raise SystemExit(f"missing target column: {args.target}")

    # Exclude the target from the feature matrix (leakage guard).
    X = _select_features(df, target=args.target)
    y = df[args.target]

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) == 0:
        raise SystemExit("no valid rows after cleaning; check input features/target")

    task = args.task or _auto_task(y)
    df_masked = df.loc[mask].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    folds, split_reports, model = [], [], None
    for i in range(args.n_repeats):
        seed = args.seed + i
        X_tr, X_te, y_tr, y_te, report = _split_once(df_masked, X, y, task, args, seed)

        # A metric on a handful of rows is noise; R2 is undefined at n == 1.
        if len(X_te) < MIN_TEST_ROWS:
            raise SystemExit(
                f"test set has {len(X_te)} rows (minimum {MIN_TEST_ROWS}); "
                f"metrics would be meaningless. Use more data or a larger --test-size."
            )
        model, m = _fit_eval(X_tr, X_te, y_tr, y_te, task, args, seed)
        m["seed"] = seed
        folds.append(m)
        if report is not None:
            split_reports.append(report)

    summary = _summarise(folds)

    outdir = Path(args.outdir) / args.target
    outdir.mkdir(parents=True, exist_ok=True)

    dump(model, outdir / "model.joblib")   # the final repeat's model
    _write_json(outdir / "features.json", {"columns": list(X.columns)})
    _write_json(outdir / "metrics.json", {"summary": summary, "folds": folds})
    _write_json(outdir / "split.json", {
        "split_requested": args.split,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_repeats": args.n_repeats,
        "n_total_rows": int(len(X)),
        "task": task,
        "scaffold_reports": split_reports or None,
    })

    print(json.dumps({"outdir": str(outdir), "summary": summary},
                     indent=2, allow_nan=False, default=_json_safe))
