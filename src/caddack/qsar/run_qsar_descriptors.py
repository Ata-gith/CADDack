#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd
from caddack.qsar.descriptors import featurize_dataframe


def add_cli(subparsers):
    p = subparsers.add_parser("qsar-descriptors", help="Featurize a CSV with SMILES")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--smiles-col", default="SMILES", help="Column containing SMILES")
    p.add_argument("--radius", type=int, default=2, help="Morgan radius")
    p.add_argument("--bits", type=int, default=2048, help="Fingerprint length")
    p.add_argument(
        "--out",
        default="data/processed/features.parquet",
        help="Output file (.parquet or .csv)",
    )
    p.add_argument(
        "--drop-errors",
        action="store_true",
        help="Drop rows where SMILES parsing failed",
    )
    p.set_defaults(func=run)


def run(args):
    inp = Path(args.csv)
    if not inp.exists():
        sys.stderr.write(f"Input not found: {inp}\n")
        sys.exit(2)

    df = pd.read_csv(inp)
    feats = featurize_dataframe(df, smiles_col=args.smiles_col, radius=args.radius, n_bits=args.bits)

    if args.drop_errors and "__error" in feats.columns:
        feats = feats[feats["__error"] != "invalid_smiles"]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if outp.suffix.lower() == ".csv":
        feats.to_csv(outp, index=False)
    else:
        # default to parquet
        try:
            feats.to_parquet(outp, index=False)
        except Exception as e:
            # fallback to csv if pyarrow/fastparquet missing
            fallback = outp.with_suffix(".csv")
            sys.stderr.write(f"Parquet write failed ({e}); writing CSV: {fallback}\n")
            feats.to_csv(fallback, index=False)
            outp = fallback

    # minimal stdout report
    n_err = int((feats["__error"] == "invalid_smiles").sum()) if "__error" in feats.columns else 0
    print(f"wrote={outp} rows={len(feats)} errors={n_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
