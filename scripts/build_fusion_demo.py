#!/usr/bin/env python3
"""Train the two-tower Bayesian affinity model and export the fusion demo page's data.

Produces ``docs/fusion.json``: per-complex predictions with their uncertainty
decomposition on a held-out scaffold split, plus 3D coordinates for a subset of
complexes so the page can show the actual binding site.

    # one-time, ~2 GB download and a few minutes of parsing
    python scripts/benchmark_pdbbind.py --data-dir /path/to/pdbbind --download --build

    python scripts/build_fusion_demo.py --data-dir /path/to/pdbbind --epochs 25

Unlike the solubility demo, this model is *not* evaluated in the browser. It
needs a protein-ligand complex with 3D coordinates, which a visitor cannot
supply by typing, and the Bayesian head needs Monte Carlo sampling over weight
posteriors. So the page is an explorer over precomputed predictions, and it says
so. What it shows that a static table could not: whether the uncertainty the
model reports actually tracks the errors it makes.

Training resumes from ``<prefix>_ckpt.pt`` so a long run can be interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_NAME = "pdbbind_examples.pkl"


def build_pairs(examples, y_mean, y_std):
    """Tensorise complexes, keeping examples and pairs aligned.

    The benchmark script drops unparseable rows without recording which, which
    is fine when only aggregate metrics are wanted. Here every prediction has to
    be attributable to a named PDB entry, so the surviving examples come back
    alongside their tensors.
    """
    import torch

    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.train import _build_geo_pyg

    pairs, kept = [], []
    for ex in examples:
        try:
            lig = to_pyg_data(smiles_to_graph_arrays(ex.ligand_smiles), y=ex.affinity)
            geo = _build_geo_pyg(ex)
            geo.y = torch.tensor([(float(ex.affinity) - y_mean) / y_std], dtype=torch.float)
        except Exception:
            continue
        pairs.append((lig, geo))
        kept.append(ex)
    return pairs, kept


def structure_payload(ex, decimals: int = 2) -> dict:
    """Coordinates for the 3D viewer: ligand atoms first, then pocket."""
    Z_SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
                14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 25: "Mn",
                26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 34: "Se",
                35: "Br", 53: "I"}
    geo = ex.geo
    xyz = []
    for (x, y, z) in geo.positions:
        xyz += [round(float(x), decimals), round(float(y), decimals), round(float(z), decimals)]
    return {
        "elements": [Z_SYMBOL.get(z, "C") for z in geo.atomic_nums],
        "xyz": xyz,
        "n_ligand": geo.n_ligand,
        "smiles": ex.ligand_smiles,
    }


def pick_showcase(records, n: int) -> list[str]:
    """Choose complexes spanning the behaviour worth looking at.

    A random sample would mostly show typical cases. The interesting ones are the
    extremes: confident and right, confident and wrong, and openly uncertain.
    """
    by = {r["id"]: r for r in records}
    err = sorted(records, key=lambda r: abs(r["pred"] - r["measured"]))
    unc = sorted(records, key=lambda r: r["epi"] + r["ale"])
    k = max(1, n // 4)
    chosen: list[str] = []
    for group in (err[:k],                      # most accurate
                  err[-k:],                     # worst misses
                  unc[:k],                      # model claims confidence
                  unc[-k:]):                    # model admits doubt
        for r in group:
            if r["id"] not in chosen:
                chosen.append(r["id"])
    return [c for c in chosen if c in by][:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory holding the cached pdbbind_examples.pkl")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "fusion.json")
    ap.add_argument("--prefix", default="fusion_demo")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    # The N << P regime makes the KL term dominate the likelihood; tempering it
    # was worth most of the achievable R2 on this split.
    ap.add_argument("--kl-weight", type=float, default=0.01)
    ap.add_argument("--kl-warmup", type=int, default=5)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-structures", type=int, default=28,
                    help="complexes shipped with 3D coordinates")
    args = ap.parse_args(argv)

    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

    from caddack.gnn.bayes import elbo_loss
    from caddack.gnn.models import FusionAffinityNet
    from caddack.qsar.split import scaffold_split

    torch.manual_seed(args.seed)
    with open(args.data_dir / CACHE_NAME, "rb") as fh:
        examples = pickle.load(fh)
    examples = [e for e in examples if getattr(e, "is_refined", True)]
    print(f"  refined complexes: {len(examples)}")

    # Scaffold split on the ligands: test ligands are chemically novel, which is
    # the honest test for a model meant to generalise to new chemistry.
    frame = pd.DataFrame({"smiles": [e.ligand_smiles for e in examples]})
    tr_idx, te_idx, report = scaffold_split(frame, smiles_col="smiles",
                                            test_size=args.test_size,
                                            seed=args.seed, return_report=True)
    train_ex = [examples[i] for i in tr_idx]
    test_ex = [examples[i] for i in te_idx]
    print(f"  scaffold split: train={len(train_ex)} test={len(test_ex)} "
          f"achieved={report.achieved_test_size:.3f} groups={report.n_scaffold_groups}")

    # Train-only statistics; standardising removes the offset the zero-mean
    # Bayesian prior otherwise leaves in the predictions.
    train_aff = np.array([e.affinity for e in train_ex], dtype=float)
    y_mean, y_std = float(train_aff.mean()), float(train_aff.std(ddof=1)) or 1.0
    print(f"  target standardisation: mean={y_mean:.3f} std={y_std:.3f}")

    train_pairs, _ = build_pairs(train_ex, y_mean, y_std)
    test_pairs, test_kept = build_pairs(test_ex, y_mean, y_std)
    n_train = len(train_pairs)
    print(f"  tensorised: train={n_train} test={len(test_pairs)}")

    def collate(batch):
        ligs, geos = zip(*batch)
        return Batch.from_data_list(list(ligs)), Batch.from_data_list(list(geos))

    train_loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_pairs, batch_size=64, shuffle=False, collate_fn=collate)

    model = FusionAffinityNet.build(
        ligand_in_channels=7, ligand_edge_dim=7, hidden_channels=args.hidden,
        num_gine_layers=3, num_geo_interactions=2, num_rbf=24, cutoff=6.0,
        bayesian_hidden=[256, 128], prior_sigma=args.prior_sigma, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt, best_path = f"{args.prefix}_ckpt.pt", f"{args.prefix}_best.pt"
    start_epoch, best_loss = 0, float("inf")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch, best_loss = state["epoch"], state.get("best_loss", float("inf"))
        print(f"  resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        beta = args.kl_weight * min(1.0, (epoch + 1) / max(args.kl_warmup, 1))
        for lig_b, geo_b in train_loader:
            opt.zero_grad()
            mu, log_var = model(lig_b, geo_b)
            loss = elbo_loss(mu, log_var, geo_b.y.view(-1), model.kl(),
                             n_train=n_train, kl_weight=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss={avg:.4f}  beta={beta:.3f}"
              f"  ({time.perf_counter()-t0:.0f}s)", flush=True)
        if beta >= args.kl_weight and avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "best_loss": best_loss}, ckpt)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()

    preds, epis, ales, truth = [], [], [], []
    with torch.no_grad():
        for lig_b, geo_b in test_loader:
            m, e, a = model.predict_with_uncertainty(lig_b, geo_b, n_samples=args.mc_samples)
            preds += m.tolist()
            epis += e.tolist()
            ales += a.tolist()
            truth += geo_b.y.view(-1).tolist()

    # Back to affinity units. Uncertainties scale but carry no offset.
    preds = np.array(preds) * y_std + y_mean
    truth = np.array(truth) * y_std + y_mean
    epis, ales = np.array(epis) * y_std, np.array(ales) * y_std
    total_sd = np.sqrt(epis ** 2 + ales ** 2)

    assert len(preds) == len(test_kept), "predictions and complexes fell out of step"

    records = [{"id": ex.pdb_id,
                "measured": round(float(t), 3),
                "pred": round(float(p), 3),
                "epi": round(float(e), 3),
                "ale": round(float(a), 3),
                "n_ligand": ex.geo.n_ligand,
                "n_pocket": ex.geo.n_pocket}
               for ex, p, t, e, a in zip(test_kept, preds, truth, epis, ales)]

    metrics = {
        "n_train": n_train,
        "n_test": len(preds),
        "mae": float(np.mean(np.abs(preds - truth))),
        "rmse": float(np.sqrt(np.mean((preds - truth) ** 2))),
        "r2": float(1 - np.sum((preds - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "pearson": float(np.corrcoef(preds, truth)[0, 1]),
        "baseline_mae": float(np.mean(np.abs(y_mean - truth))),
        "coverage_1sigma": float(np.mean(np.abs(preds - truth) <= total_sd)),
        "coverage_2sigma": float(np.mean(np.abs(preds - truth) <= 2 * total_sd)),
    }
    print("  test:", {k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})

    # Does claimed uncertainty track actual error? Binned, so the page can show it.
    order = np.argsort(total_sd)
    bins = np.array_split(order, 5)
    reliability = [{"mean_uncertainty": round(float(total_sd[b].mean()), 3),
                    "mean_abs_error": round(float(np.abs(preds[b] - truth[b]).mean()), 3),
                    "n": int(len(b))} for b in bins]
    print("  uncertainty vs error by quintile:",
          [(r["mean_uncertainty"], r["mean_abs_error"]) for r in reliability])

    showcase = pick_showcase(records, args.n_structures)
    by_id = {ex.pdb_id: ex for ex in test_kept}
    structures = {pid: structure_payload(by_id[pid]) for pid in showcase if pid in by_id}

    bundle = {"metrics": metrics, "split": report.as_dict(), "complexes": records,
              "reliability": reliability, "structures": structures,
              "target": "-log Kd/Ki (pK units)",
              "dataset": "PDBbind v2015 refined set",
              "config": {"epochs": args.epochs, "kl_weight": args.kl_weight,
                         "mc_samples": args.mc_samples, "hidden": args.hidden}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    print(f"  wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB, "
          f"{len(structures)} structures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
