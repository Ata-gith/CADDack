#!/usr/bin/env python3
"""Does a deep ensemble produce a better error signal than the Bayesian head?

The fusion model's epistemic term — sampled from a mean-field Gaussian posterior
around one solution — ranks its own errors at rho = -0.069, i.e. slightly the
wrong way, while a model-free Tanimoto distance to the training ligands manages
+0.093. One explanation is that sampling perturbs a single solution rather than
exploring genuinely different ones: all the samples agree, and on an unfamiliar
complex they are confidently wrong together.

A deep ensemble tests that. Train the same architecture several times from
different initialisations on the same split, then read the spread *across models*
as the epistemic term. Where independently-trained networks disagree is where the
data did not pin the answer down (Lakshminarayanan et al., NeurIPS 2017).

The decomposition is unchanged from the single-model case — mean of the means,
variance of the means, mean of the predicted variances — only the source of the
samples differs.

    python scripts/ensemble_eval.py --data-dir /path/to/pdbbind \\
        --weights ens0_best.pt ens1_best.pt ens2_best.pt ens3_best.pt ens4_best.pt

Members must share a split: train them with the same --seed and different
--init-seed, or they are scored on different test sets.
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
    return None if mol is None else np.unpackbits(mol_to_ecfp_array(mol, n_bits=n_bits))


def regression_metrics(pred, truth) -> dict:
    return {
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "r2": float(1 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "pearson": float(np.corrcoef(pred, truth)[0, 1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, nargs="+", required=True,
                    help="one checkpoint per ensemble member")
    ap.add_argument("--mc-samples", type=int, default=30,
                    help="MC passes for the single-model comparison")
    ap.add_argument("--n-bits", type=int, default=1024)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="must match the members' split seed")
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

    def fresh_model(path):
        m = FusionAffinityNet.build(
            ligand_in_channels=7, ligand_edge_dim=7, hidden_channels=128,
            num_gine_layers=3, num_geo_interactions=2, num_rbf=24, cutoff=6.0,
            geo_node_feat_dim=NEIGHBOR_FEAT_DIM, bayesian_hidden=[256, 128], dropout=0.1)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m

    # Each member contributes one deterministic prediction and one predicted
    # variance. Dropout stays off: the spread we want is between models.
    member_mu, member_var, truth = [], [], None
    for path in args.weights:
        model = fresh_model(path)
        mus, variances, ys = [], [], []
        with torch.no_grad():
            for lig_b, geo_b in loader:
                mu, log_var = model(lig_b, geo_b)
                mus += mu.tolist()
                variances += torch.exp(log_var).tolist()
                ys += geo_b.y.view(-1).tolist()
        member_mu.append(np.array(mus))
        member_var.append(np.array(variances))
        truth = np.array(ys)
        print(f"  loaded {path.name}")

    mu_stack = np.stack(member_mu)                      # [M, N]
    var_stack = np.stack(member_var)

    ens_pred = mu_stack.mean(0) * y_std + y_mean
    ens_epi = mu_stack.std(0, ddof=0) * y_std           # disagreement between models
    ens_ale = np.sqrt(var_stack.mean(0)) * y_std
    truth_pk = truth * y_std + y_mean
    err = np.abs(ens_pred - truth_pk)

    # The single-model baseline, on the same test set, for a like-for-like read.
    single = fresh_model(args.weights[0])
    torch.manual_seed(args.seed)
    s_pred, s_epi, s_ale = [], [], []
    with torch.no_grad():
        for lig_b, geo_b in loader:
            m_, e_, a_ = single.predict_with_uncertainty(
                lig_b, geo_b, n_samples=args.mc_samples, mc_dropout=True)
            s_pred += m_.tolist()
            s_epi += e_.tolist()
            s_ale += a_.tolist()
    s_pred = np.array(s_pred) * y_std + y_mean
    s_epi, s_ale = np.array(s_epi) * y_std, np.array(s_ale) * y_std
    s_err = np.abs(s_pred - truth_pk)

    train_fps = [f for f in (fingerprint(e.ligand_smiles, args.n_bits) for e in train_ex)
                 if f is not None]
    ad = SimilarityAD.fit(np.stack(train_fps))
    ad_dist = np.array([1.0 - float(ad.similarity(
        fingerprint(e.ligand_smiles, args.n_bits)[None, :])[0]) for e in kept])

    print(f"\n  {len(args.weights)} members, {len(kept)} held-out complexes\n")
    print("  accuracy")
    for label, pred in [("single model (MC dropout)", s_pred), ("ensemble mean", ens_pred)]:
        m = regression_metrics(pred, truth_pk)
        print(f"    {label:<30} MAE {m['mae']:.3f}  R2 {m['r2']:+.3f}  r {m['pearson']:+.3f}")

    print("\n  does the signal rank the errors it was computed against?")
    rows = [
        ("single: epistemic (MC sampling)", s_epi, s_err),
        ("single: total sigma", np.hypot(s_epi, s_ale), s_err),
        ("ensemble: epistemic (disagreement)", ens_epi, err),
        ("ensemble: total sigma", np.hypot(ens_epi, ens_ale), err),
        ("AD distance (model-free)", ad_dist, err),
        ("ensemble total + AD, rank-averaged",
         rankdata(np.hypot(ens_epi, ens_ale)) + rankdata(ad_dist), err),
    ]
    for label, signal, target in rows:
        print(f"    {label:<38} rho {spearmanr(signal, target).statistic:+.3f}")

    q = np.array_split(np.argsort(ens_epi), 5)
    print("\n  ensemble disagreement vs error, by quintile")
    for i, b in enumerate(q):
        print(f"    Q{i+1}  mean disagreement {ens_epi[b].mean():.3f}   MAE {err[b].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
