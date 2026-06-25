"""
Example 05: Two-Tower Bayesian Fusion Model
============================================
Demonstrates FusionAffinityNet — a Bayesian scoring function that fuses
a 2D ligand GNN (Tower 1) with a 3D geometry tower (Tower 2) and outputs
binding affinity predictions with calibrated uncertainty estimates.

Uncertainty is decomposed into:
  - Epistemic (model) uncertainty  — decreases with more training data
  - Aleatoric (data) uncertainty   — irreducible noise in the measurements

No real PDB/SDF files are needed; synthetic ComplexExample objects are
built inline, matching the pattern used in the test suite.

Requires: rdkit + torch + torch-geometric + torch-scatter + scikit-learn
  conda install -c conda-forge rdkit
  pip install torch torch-geometric torch-scatter scikit-learn
"""

import sys
import json
import tempfile

try:
    import torch
    from torch_geometric.data import Data, Batch

    from caddack.gnn.geometry import GeometryRecord, ComplexExample
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet
    from caddack.gnn.train import train_fusion_from_complexes
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:")
    print("  conda install -c conda-forge rdkit")
    print("  pip install torch torch-geometric torch-scatter scikit-learn")
    sys.exit(1)

# ── 1. Build synthetic ComplexExample objects ─────────────────────────────────

def make_complex(smiles: str, affinity: float, n_lig: int = 8, n_pocket: int = 20) -> ComplexExample:
    """Synthetic complex: atoms placed along the x-axis, no real structure."""
    n_total = n_lig + n_pocket
    geo = GeometryRecord(
        atomic_nums=[6] * n_total,
        positions=[(float(i) * 0.5, 0.0, 0.0) for i in range(n_total)],
        n_ligand=n_lig,
        n_pocket=n_pocket,
    )
    return ComplexExample(pdb_id="SYN", affinity=affinity, ligand_smiles=smiles, geo=geo)

complexes = [
    make_complex("CC(=O)Oc1ccccc1C(=O)O",       affinity=5.2),   # aspirin
    make_complex("CC(C)Cc1ccc(cc1)C(C)C(=O)O",  affinity=6.1),   # ibuprofen
    make_complex("CC(=O)Nc1ccc(O)cc1",           affinity=4.8),   # paracetamol
    make_complex("Cn1cnc2c1c(=O)n(c(=O)n2C)C",  affinity=3.9),   # caffeine
    make_complex("c1ccc(cc1)C(=O)O",             affinity=4.3),   # benzoic acid
    make_complex("OC(=O)c1ccccc1O",              affinity=4.5),   # salicylic acid
    make_complex("COc1ccc2cc(ccc2c1)C(C)C(=O)O",affinity=6.4),   # naproxen
    make_complex("CC(=O)Nc1ccc(cc1)S(N)(=O)=O", affinity=6.8),   # sulfacetamide
]

print(f"Synthetic dataset: {len(complexes)} complexes")
print()

# ── 2. Build the model ────────────────────────────────────────────────────────

print("── Building FusionAffinityNet ───────────────────────────")
model = FusionAffinityNet.build(
    ligand_in_channels=7,
    ligand_edge_dim=7,
    hidden_channels=64,
    num_gine_layers=2,
    num_geo_interactions=2,
    num_rbf=20,
    cutoff=4.0,
    bayesian_hidden=[128, 64],
    prior_sigma=1.0,
    dropout=0.0,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
print()

# ── 3. Single forward pass → (mu, log_var) ───────────────────────────────────

print("── Single forward pass ──────────────────────────────────")

def build_lig(ex: ComplexExample):
    g = smiles_to_graph_arrays(ex.ligand_smiles)
    return to_pyg_data(g)

def build_geo(ex: ComplexExample):
    geo = ex.geo
    z = torch.tensor(geo.atomic_nums, dtype=torch.long)
    pos = torch.tensor(geo.positions, dtype=torch.float)
    return Data(z=z, pos=pos)

ex = complexes[0]
lig_data = Batch.from_data_list([build_lig(ex)])
geo_data = Batch.from_data_list([build_geo(ex)])

model.eval()
with torch.no_grad():
    mu, log_var = model(lig_data, geo_data)

print(f"mu (predicted affinity)  : {mu.item():.4f}")
print(f"log_var (log aleatoric)  : {log_var.item():.4f}")
print(f"sigma (aleatoric noise)  : {log_var.exp().sqrt().item():.4f}")
print()

# ── 4. Uncertainty estimation via MC sampling ─────────────────────────────────

print("── Uncertainty via MC sampling (30 weight samples) ──────")
pred_mean, epistemic_std, aleatoric_std = model.predict_with_uncertainty(
    lig_data, geo_data, n_samples=30
)

print(f"Predicted affinity       : {pred_mean.item():.4f}")
print(f"Epistemic std (model)    : {epistemic_std.item():.4f}  ← decreases with more data")
print(f"Aleatoric std (noise)    : {aleatoric_std.item():.4f}  ← irreducible data noise")
print(f"Total uncertainty        : {(epistemic_std + aleatoric_std).item():.4f}")
print()

# ── 5. Quick training run ─────────────────────────────────────────────────────

print("── Training for 5 epochs (synthetic data) ───────────────")
with tempfile.TemporaryDirectory() as outdir:
    metrics = train_fusion_from_complexes(
        complexes=complexes,
        outdir=outdir,
        hidden_channels=64,
        num_gine_layers=2,
        num_geo_interactions=2,
        num_rbf=20,
        cutoff=4.0,
        bayesian_hidden=[128, 64],
        prior_sigma=1.0,
        kl_weight=0.1,
        kl_warmup=3,
        epochs=5,
        batch_size=4,
        lr=1e-3,
        test_size=0.25,
        split="random",
        mc_samples_eval=10,
    )
    print("Metrics:", json.dumps(metrics, indent=2))

print()
print("For real use, pass a PDBbind-style dataset via load_complex_dataset()")
print("and point train_fusion_from_complexes at the resulting ComplexExample list.")
