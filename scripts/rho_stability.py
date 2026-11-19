#!/usr/bin/env python3
"""Is the fusion page's signal comparison stable, or an artefact of one MC pass?

``docs/fusion.html`` ranks several uncertainty signals by how well each predicts
the model's errors. Those correlations come from one Monte Carlo pass, and
dropout makes every pass different -- re-exporting the same trained model once
moved the combined rho from 0.146 to 0.112, a swing as large as the gaps being
compared. So the ranking needed checking before anything was claimed from it.

This repeats the comparison over several passes of the same weights. Measured
over eight, on the PDBbind scaffold split:

    signal                      mean      sd     range
    epistemic sigma           -0.069   0.016   -0.094 .. -0.048
    aleatoric sigma           +0.089   0.013   +0.078 .. +0.118
    both, as reported         +0.087   0.022   +0.063 .. +0.134
    AD distance               +0.093   0.004   +0.089 .. +0.098
    combined                  +0.128   0.017   +0.101 .. +0.159

Two orderings held in every pass: the epistemic term never crossed zero, and the
combination beat each of its parts 8/8. The middle three sit within noise of one
another and should not be ranked. The applicability-domain signal is also the
most reproducible by a wide margin, since it involves no sampling at all -- for
deciding whether to trust a prediction, that matters as much as the correlation.

Needs a checkpoint from scripts/build_fusion_demo.py:

    python scripts/build_fusion_demo.py --data-dir /path/to/pdbbind --prefix fusion_v3
    python scripts/rho_stability.py --data-dir /path/to/pdbbind --weights fusion_v3_best.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caddack.qsar.applicability import SimilarityAD  # noqa: E402
from caddack.qsar.descriptors import mol_to_ecfp_array, parse_smiles  # noqa: E402
from caddack.qsar.split import scaffold_split  # noqa: E402

CACHE_NAME = "pdbbind_examples.pkl"


def fingerprint(smiles: str, n_bits: int):
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return np.unpackbits(mol_to_ecfp_array(mol, n_bits=n_bits))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=REPO_ROOT / "fusion_v3_best.pt")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--n-bits", type=int, default=1024)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.geometry import NEIGHBOR_FEAT_DIM
    from caddack.gnn.models import FusionAffinityNet
    from caddack.gnn.train import _build_geo_pyg

    with open(args.data_dir / CACHE_NAME, "rb") as fh:
        examples = [e for e in pickle.load(fh) if getattr(e, "is_refined", True)]

    # The same deterministic split the exporter uses, so the test set matches.
    frame = pd.DataFrame({"smiles": [e.ligand_smiles for e in examples]})
    train_idx, test_idx = scaffold_split(frame, smiles_col="smiles",
                                         test_size=args.test_size, seed=args.seed)
    train_ex = [examples[i] for i in train_idx]
    test_ex = [examples[i] for i in test_idx]

    affinities = np.array([e.affinity for e in train_ex], dtype=float)
    y_mean, y_std = affinities.mean(), affinities.std(ddof=1)

    pairs, kept = [], []
    for ex in test_ex:
        try:
            lig = to_pyg_data(smiles_to_graph_arrays(ex.ligand_smiles), y=ex.affinity)
            geo = _build_geo_pyg(ex, with_neighbors=True)
            geo.y = torch.tensor([(float(ex.affinity) - y_mean) / y_std], dtype=torch.float)
        except Exception:
            continue
        pairs.append((lig, geo))
        kept.append(ex)

    def collate(batch):
        ligs, geos = zip(*batch)
        return Batch.from_data_list(list(ligs)), Batch.from_data_list(list(geos))

    loader = DataLoader(pairs, batch_size=64, shuffle=False, collate_fn=collate)

    model = FusionAffinityNet.build(
        ligand_in_channels=7, ligand_edge_dim=7, hidden_channels=128,
        num_gine_layers=3, num_geo_interactions=2, num_rbf=24, cutoff=6.0,
        geo_node_feat_dim=NEIGHBOR_FEAT_DIM, bayesian_hidden=[256, 128], dropout=0.1)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # Deterministic, so it contributes no variance of its own across passes.
    train_fps = [f for f in (fingerprint(e.ligand_smiles, args.n_bits) for e in train_ex)
                 if f is not None]
    ad = SimilarityAD.fit(np.stack(train_fps))
    ad_distance = np.array([
        1.0 - float(ad.similarity(fingerprint(e.ligand_smiles, args.n_bits)[None, :])[0])
        for e in kept
    ])

    results: dict[str, list[float]] = {k: [] for k in
                                       ("epi", "ale", "model", "ad", "comb")}
    for seed in range(args.passes):
        torch.manual_seed(seed)
        preds, epis, ales, truth = [], [], [], []
        with torch.no_grad():
            for lig_b, geo_b in loader:
                mu, e, a = model.predict_with_uncertainty(
                    lig_b, geo_b, n_samples=args.mc_samples, mc_dropout=True)
                preds += mu.tolist()
                epis += e.tolist()
                ales += a.tolist()
                truth += geo_b.y.view(-1).tolist()

        preds = np.array(preds) * y_std + y_mean
        truth = np.array(truth) * y_std + y_mean
        epis, ales = np.array(epis) * y_std, np.array(ales) * y_std
        err = np.abs(preds - truth)
        total = np.hypot(epis, ales)

        results["epi"].append(spearmanr(epis, err).statistic)
        results["ale"].append(spearmanr(ales, err).statistic)
        results["model"].append(spearmanr(total, err).statistic)
        results["ad"].append(spearmanr(ad_distance, err).statistic)
        results["comb"].append(
            spearmanr(rankdata(total) + rankdata(ad_distance), err).statistic)

    print(f"  {len(kept)} complexes, {args.passes} independent MC passes "
          f"of the same trained model\n")
    print(f"  {'signal':<34}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}")
    labels = [("epi", "epistemic sigma"), ("ale", "aleatoric sigma"),
              ("model", "model total sigma"), ("ad", "AD distance (deterministic)"),
              ("comb", "combined")]
    for key, label in labels:
        v = np.array(results[key])
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"  {label:<34}{v.mean():>+8.3f}{sd:>8.3f}{v.min():>+8.3f}{v.max():>+8.3f}")

    comb = np.array(results["comb"])
    print(f"\n  combined beat model-only in {int((comb > np.array(results['model'])).sum())}"
          f"/{args.passes} passes")
    print(f"  combined beat AD-only      in {int((comb > np.array(results['ad'])).sum())}"
          f"/{args.passes} passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
