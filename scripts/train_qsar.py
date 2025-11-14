#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from joblib import dump

from caddack.qsar.split import scaffold_split

NUM_DESC = ["MolWt","LogP","TPSA","NumHBD","NumHBA","NumRotBonds"]


def add_cli(subparsers):
    p = subparsers.add_parser("train-qsar", help="Train baseline QSAR on Parquet features")
    p.add_argument("--parquet", required=True, help="Input features parquet")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--task", choices=["regression","classification"], default=None, help="If omitted, auto-detect")
    p.add_argument("--split", choices=["random","scaffold"], default="scaffold")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="models/qsar", help="Output directory")
    p.add_argument("--max-features", type=str, default="auto", help="RandomForest max_features")
    p.add_argument("--n-estimators", type=int, default=300)
    p.set_defaults(func=run)


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    bits = [c for c in df.columns if c.startswith("ECFP")]
    cols = [c for c in NUM_DESC if c in df.columns] + bits
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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def _metrics_clf(y_true, y_prob, y_pred) -> dict:
    out = {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }
    return out


def run(args):
    inp = Path(args.parquet)
    df0 = pd.read_parquet(inp)
    df = _drop_invalid(df0)

    if args.target not in df.columns:
        raise SystemExit(f"missing target column: {args.target}")

    # select features and target
    X = _select_features(df)
    y = df[args.target]

    # replace infinities and drop NaNs only where they matter
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) == 0:
        raise SystemExit("no valid rows after cleaning; check input features/target")

    task = args.task or _auto_task(y)

    if args.split == "scaffold":
        if "SMILES_canonical" not in df.columns:
            raise SystemExit("SMILES_canonical required for scaffold split")
        train_idx, test_idx = scaffold_split(
            df.loc[mask],  # use the same filtered set
            smiles_col="SMILES_canonical",
            test_size=args.test_size,
            seed=args.seed,
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=(y if task == "classification" else None),
        )

    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            # "auto" in sklearn is deprecated; map it to default behavior
            max_features="sqrt" if args.max_features == "auto" else args.max_features,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = _metrics_reg(y_test, y_pred)

    else:
        # classification
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # guard: need at least 2 classes after split
        uniq = np.unique(y_train)
        if uniq.size < 2:
            raise SystemExit(
                f"classification task but only one class present in training data: {uniq.tolist()}"
            )

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced",
            # for classifier, sklearn default is "sqrt"
            max_features="sqrt" if args.max_features == "auto" else args.max_features,
        )
        model.fit(X_train, y_train)

        # select probability column corresponding to label 1 if present,
        # otherwise use the second class as "positive" by convention
        classes = model.classes_
        if classes.shape[0] != 2:
            raise SystemExit(f"expected binary classification, got classes={classes.tolist()}")

        if 1 in classes:
            pos_idx = int(np.where(classes == 1)[0][0])
        else:
            pos_idx = 1  # classes are sorted; take the second as "positive"

        y_prob = model.predict_proba(X_test)[:, pos_idx]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _metrics_clf(y_test, y_prob, y_pred)


    outdir = Path(args.outdir) / args.target
    outdir.mkdir(parents=True, exist_ok=True)

    dump(model, outdir / "model.joblib")
    (outdir / "features.json").write_text(
        json.dumps({"columns": list(X.columns)}, indent=2),
        encoding="utf-8",
    )
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (outdir / "split.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "test_size": args.test_size,
                "seed": args.seed,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "task": task,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"outdir": str(outdir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
