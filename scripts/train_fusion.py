#!/usr/bin/env python3
"""train-fusion: train the two-tower Bayesian affinity model on a PDBbind-style dataset."""
import argparse
import json
import sys
from pathlib import Path


def add_cli(subparsers):
    p = subparsers.add_parser("train-fusion", help="Train two-tower Bayesian affinity model")
    p.add_argument("--root", required=True,
                   help="Root directory of the PDBbind-style dataset")
    p.add_argument("--index-csv", required=True,
                   help="CSV with columns: pdb_id, affinity (and optionally smiles)")
    p.add_argument("--target", default="affinity",
                   help="Column name for the affinity target (default: affinity)")
    p.add_argument("--smiles-col", default="smiles",
                   help="Column name for SMILES (optional; used for scaffold split)")
    p.add_argument("--pdb-col", default="pdb_id", help="Column name for PDB IDs")
    p.add_argument("--ligand-suffix", default=".sdf",
                   help="Ligand file suffix (.sdf or .mol2; default: .sdf)")
    p.add_argument("--cutoff", type=float, default=6.0,
                   help="Pocket extraction cutoff in Å (default: 6.0)")
    p.add_argument("--hidden", type=int, default=128,
                   help="Hidden channels for both towers (default: 128)")
    p.add_argument("--gine-layers", type=int, default=3)
    p.add_argument("--geo-interactions", type=int, default=3)
    p.add_argument("--num-rbf", type=int, default=50)
    p.add_argument("--prior-sigma", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-warmup", type=int, default=10,
                   help="Epochs over which KL weight linearly ramps up (default: 10)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--mc-samples-eval", type=int, default=30)
    p.add_argument("--no-aleatoric", action="store_true",
                   help="Use MSE loss instead of heteroscedastic NLL")
    p.add_argument("--outdir", default="models/fusion",
                   help="Output directory for model.pt / metrics.json / config.json")
    p.set_defaults(func=run)


def run(args):
    from caddack.gnn.geometry import load_complex_dataset
    from caddack.gnn.train import train_fusion_from_complexes

    sys.stderr.write(f"Loading complexes from {args.root} ...\n")
    complexes = load_complex_dataset(
        index_csv=args.index_csv,
        root=args.root,
        smiles_col=args.smiles_col,
        affinity_col=args.target,
        pdb_col=args.pdb_col,
        ligand_suffix=args.ligand_suffix,
        cutoff=args.cutoff,
    )
    sys.stderr.write(f"Loaded {len(complexes)} valid complexes.\n")

    metrics = train_fusion_from_complexes(
        complexes=complexes,
        outdir=args.outdir,
        hidden_channels=args.hidden,
        num_gine_layers=args.gine_layers,
        num_geo_interactions=args.geo_interactions,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        prior_sigma=args.prior_sigma,
        kl_weight=args.kl_weight,
        kl_warmup=args.kl_warmup,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        split=args.split,
        mc_samples_eval=args.mc_samples_eval,
        no_aleatoric=args.no_aleatoric,
    )

    result = {"outdir": str(Path(args.outdir).resolve()), "metrics": metrics}
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    parsed = parser.parse_args()
    parsed.func(parsed)
