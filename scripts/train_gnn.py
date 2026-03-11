#!/usr/bin/env python3
import argparse
import json

from caddack.gnn.train import train_from_csv


def add_cli(subparsers):
    p = subparsers.add_parser("train-gnn", help="Train molecular GNN on SMILES + target CSV")
    p.add_argument("--csv", required=True, help="Input CSV file")
    p.add_argument("--smiles-col", default="SMILES", help="SMILES column")
    p.add_argument("--target", required=True, help="Target column")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--model", choices=["gine", "gcn"], default="gine")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="models/gnn")
    p.set_defaults(func=run)


def run(args):
    metrics = train_from_csv(
        csv_path=args.csv,
        smiles_col=args.smiles_col,
        target_col=args.target,
        outdir=args.outdir,
        model_name=args.model,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps({"outdir": args.outdir, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
