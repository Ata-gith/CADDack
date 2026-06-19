# tests/test_fusion.py
"""Tests for the two-tower Bayesian fusion architecture.

No-dep tests (PDB parser, import guards) run without any optional packages.
BNN / forward / training tests are gated on torch + torch-geometric + rdkit.
"""
import importlib.util
import types

import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None
torch_available = importlib.util.find_spec("torch") is not None
pyg_available = importlib.util.find_spec("torch_geometric") is not None
scatter_available = importlib.util.find_spec("torch_scatter") is not None

all_gnn_deps = rdkit_available and torch_available and pyg_available and scatter_available


# ---------------------------------------------------------------------------
# PDB parser — no deps needed
# ---------------------------------------------------------------------------

MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
HETATM    3  C1  LIG L   1       5.000   6.000   7.000  1.00  0.00           C
"""

MULTI_MODEL_PDB = """\
MODEL        1
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      2  CA  ALA A   1       9.000   9.000   9.000  1.00  0.00           C
ENDMDL
"""


def test_parse_pdb_atoms_basic(tmp_path):
    from caddack.gnn.geometry import parse_pdb_atoms
    pdb = tmp_path / "test.pdb"
    pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    atoms = parse_pdb_atoms(pdb)
    assert len(atoms) == 3
    assert atoms[0].record == "ATOM"
    assert atoms[0].element == "N"
    assert atoms[0].atomic_num == 7
    assert atoms[2].record == "HETATM"
    assert abs(atoms[2].x - 5.0) < 1e-6


def test_parse_pdb_atoms_missing_file(tmp_path):
    from caddack.gnn.geometry import parse_pdb_atoms
    atoms = parse_pdb_atoms(tmp_path / "nonexistent.pdb")
    assert atoms == []


def test_parse_pdb_atoms_first_model_only(tmp_path):
    from caddack.gnn.geometry import parse_pdb_atoms
    pdb = tmp_path / "multi.pdb"
    pdb.write_text(MULTI_MODEL_PDB, encoding="utf-8")
    atoms = parse_pdb_atoms(pdb)
    assert len(atoms) == 1
    assert abs(atoms[0].x - 1.0) < 1e-6


def test_extract_pocket_distance(tmp_path):
    from caddack.gnn.geometry import parse_pdb_atoms, extract_pocket, PDBAtom

    pdb = tmp_path / "test.pdb"
    pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    all_atoms = parse_pdb_atoms(pdb)
    protein = [a for a in all_atoms if a.record == "ATOM"]
    ligand = [a for a in all_atoms if a.record == "HETATM"]

    # CA at (2,3,4) → ligand (5,6,7): dist≈5.2Å (inside 6Å); N at (1,2,3): dist≈6.9Å (outside)
    pocket = extract_pocket(protein, ligand, cutoff=6.0)
    assert len(pocket) == 1

    # Tight cutoff — neither protein atom should be within 0.1Å of ligand
    pocket_tight = extract_pocket(protein, ligand, cutoff=0.1)
    assert len(pocket_tight) == 0


def test_build_geo_record():
    from caddack.gnn.geometry import parse_pdb_atoms, extract_pocket, _build_geo_record, PDBAtom
    import math

    # Build synthetic atoms directly
    def make_atom(record, x, y, z, elem="C", z_num=6):
        return PDBAtom(record=record, name=elem, resname="X", chain="A",
                       resseq=1, x=x, y=y, z=z, element=elem, atomic_num=z_num)

    lig = [make_atom("HETATM", 0, 0, 0)]
    pocket = [make_atom("ATOM", 3, 0, 0), make_atom("ATOM", 0, 3, 0)]
    rec = _build_geo_record(lig, pocket)
    assert rec.n_ligand == 1
    assert rec.n_pocket == 2
    assert len(rec.atomic_nums) == 3
    assert len(rec.positions) == 3


# ---------------------------------------------------------------------------
# Import guards — require torch/rdkit to be absent to trigger; skip if present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(torch_available, reason="torch installed")
def test_bayesian_linear_requires_torch():
    from caddack.gnn.bayes import BayesianLinear
    with pytest.raises(ImportError, match="PyTorch"):
        BayesianLinear.build(4, 4)


@pytest.mark.skipif(torch_available, reason="torch installed")
def test_bayesian_mlp_requires_torch():
    from caddack.gnn.bayes import BayesianMLP
    with pytest.raises(ImportError, match="PyTorch"):
        BayesianMLP.build(4, [8])


@pytest.mark.skipif(rdkit_available, reason="rdkit installed")
def test_load_ligand_requires_rdkit(tmp_path):
    from caddack.gnn.geometry import load_ligand
    sdf = tmp_path / "mol.sdf"
    sdf.write_text("dummy", encoding="utf-8")
    with pytest.raises(ImportError, match="RDKit"):
        load_ligand(sdf)


# ---------------------------------------------------------------------------
# BNN unit tests (torch required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_bayesian_linear_forward_shape():
    import torch
    from caddack.gnn.bayes import BayesianLinear
    layer = BayesianLinear.build(in_features=8, out_features=4)
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 4)


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_bayesian_linear_kl_positive():
    import torch
    from caddack.gnn.bayes import BayesianLinear
    layer = BayesianLinear.build(in_features=8, out_features=4)
    x = torch.randn(2, 8)
    layer(x)  # forward pass triggers reparameterization
    kl = layer.kl()
    assert kl.item() >= 0.0


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_bayesian_linear_stochastic_at_train():
    """Two forward passes should give different outputs during training."""
    import torch
    from caddack.gnn.bayes import BayesianLinear
    layer = BayesianLinear.build(in_features=8, out_features=4)
    layer.train()
    x = torch.randn(2, 8)
    out1 = layer(x)
    out2 = layer(x)
    assert not torch.allclose(out1, out2), "BayesianLinear should be stochastic during training"


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_bayesian_mlp_output_shapes():
    import torch
    from caddack.gnn.bayes import BayesianMLP
    mlp = BayesianMLP.build(in_features=16, hidden_dims=[32, 16])
    mlp.train()
    x = torch.randn(5, 16)
    mu, log_var = mlp(x)
    assert mu.shape == (5,)
    assert log_var.shape == (5,)


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_bayesian_mlp_kl_accumulates():
    import torch
    from caddack.gnn.bayes import BayesianMLP
    mlp = BayesianMLP.build(in_features=8, hidden_dims=[16, 8])
    mlp.train()
    x = torch.randn(3, 8)
    mlp(x)
    kl = mlp.kl()
    assert isinstance(kl, torch.Tensor)
    assert kl.item() >= 0.0


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_elbo_loss_finite():
    import torch
    from caddack.gnn.bayes import BayesianMLP, elbo_loss
    mlp = BayesianMLP.build(in_features=4, hidden_dims=[8])
    mlp.train()
    x = torch.randn(6, 4)
    y = torch.randn(6)
    mu, log_var = mlp(x)
    kl = mlp.kl()
    loss = elbo_loss(mu, log_var, y, kl, n_train=100)
    assert torch.isfinite(loss), f"ELBO loss is not finite: {loss.item()}"


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_elbo_loss_backward():
    import torch
    from caddack.gnn.bayes import BayesianMLP, elbo_loss
    mlp = BayesianMLP.build(in_features=4, hidden_dims=[8])
    mlp.train()
    x = torch.randn(4, 4)
    y = torch.randn(4)
    mu, log_var = mlp(x)
    kl = mlp.kl()
    loss = elbo_loss(mu, log_var, y, kl, n_train=50)
    loss.backward()
    # Check that gradients flow through the Bayesian layers
    for name, p in mlp.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Geometry tower + fusion model (all GNN deps required)
# ---------------------------------------------------------------------------

def _make_synthetic_complexes(n: int = 6):
    """Create synthetic ComplexExample objects without touching the filesystem."""
    from caddack.gnn.geometry import ComplexExample, GeometryRecord

    complexes = []
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCO", "c1ccncc1"]
    for i in range(n):
        geo = GeometryRecord(
            atomic_nums=[6, 7, 8, 6, 6, 6, 8],
            positions=[
                (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0),
                (0.0, 1.5, 0.0), (1.5, 1.5, 0.0), (3.0, 1.5, 0.0),
                (0.0, 3.0, 0.0),
            ],
            n_ligand=3,
            n_pocket=4,
        )
        complexes.append(ComplexExample(
            pdb_id=f"SYNTH{i:02d}",
            affinity=float(i) * 0.5 + 4.0,
            ligand_smiles=smiles_list[i % len(smiles_list)],
            geo=geo,
        ))
    return complexes


@pytest.mark.skipif(not all_gnn_deps, reason="torch+pyg+rdkit+scatter not installed")
def test_fusion_forward_pass():
    import torch
    from torch_geometric.data import Data, Batch
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.geometry import GeometryRecord, ComplexExample
    from caddack.gnn.models import FusionAffinityNet

    # Build a minimal ligand graph
    g = smiles_to_graph_arrays("CCO")
    lig_data = to_pyg_data(g, y=5.0)
    lig_batch = Batch.from_data_list([lig_data, lig_data])

    # Build a minimal geometry graph
    z = torch.tensor([6, 7, 8, 6, 6], dtype=torch.long)
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 1.5, 0.0], [1.5, 1.5, 0.0],
    ], dtype=torch.float)
    geo_data = Data(z=z, pos=pos)
    geo_batch = Batch.from_data_list([geo_data, geo_data])

    model = FusionAffinityNet.build(
        ligand_in_channels=7,
        ligand_edge_dim=7,
        hidden_channels=32,
        num_gine_layers=2,
        num_geo_interactions=2,
        num_rbf=16,
        cutoff=5.0,
        bayesian_hidden=[32],
    )
    model.train()
    mu, log_var = model(lig_batch, geo_batch)
    assert mu.shape == (2,)
    assert log_var.shape == (2,)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_var).all()


@pytest.mark.skipif(not all_gnn_deps, reason="torch+pyg+rdkit+scatter not installed")
def test_fusion_kl_and_backward():
    import torch
    from torch_geometric.data import Data, Batch
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet
    from caddack.gnn.bayes import elbo_loss

    g = smiles_to_graph_arrays("c1ccccc1")
    lig_data = to_pyg_data(g, y=6.0)
    lig_batch = Batch.from_data_list([lig_data])

    z = torch.tensor([6, 7, 8, 6], dtype=torch.long)
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=torch.float)
    geo_data = Data(z=z, pos=pos)
    geo_batch = Batch.from_data_list([geo_data])
    geo_batch.y = torch.tensor([6.0])

    model = FusionAffinityNet.build(
        ligand_in_channels=7,
        ligand_edge_dim=7,
        hidden_channels=16,
        num_gine_layers=2,
        num_geo_interactions=2,
        num_rbf=8,
        cutoff=5.0,
        bayesian_hidden=[16],
    )
    model.train()
    mu, log_var = model(lig_batch, geo_batch)
    kl = model.kl()
    y = geo_batch.y.view(-1)
    loss = elbo_loss(mu, log_var, y, kl, n_train=10)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.skipif(not all_gnn_deps, reason="torch+pyg+rdkit+scatter not installed")
def test_fusion_uncertainty_output():
    import torch
    from torch_geometric.data import Data, Batch
    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet

    g = smiles_to_graph_arrays("CCO")
    lig_data = to_pyg_data(g, y=5.0)
    lig_batch = Batch.from_data_list([lig_data])

    z = torch.tensor([6, 7, 8], dtype=torch.long)
    pos = torch.tensor([[0, 0, 0], [1.5, 0, 0], [3, 0, 0]], dtype=torch.float)
    geo_data = Data(z=z, pos=pos)
    geo_batch = Batch.from_data_list([geo_data])

    model = FusionAffinityNet.build(
        ligand_in_channels=7, ligand_edge_dim=7,
        hidden_channels=16, num_gine_layers=2, num_geo_interactions=2,
        num_rbf=8, cutoff=5.0, bayesian_hidden=[16],
    )
    pred_mean, epistemic, aleatoric = model.predict_with_uncertainty(
        lig_batch, geo_batch, n_samples=5
    )
    assert pred_mean.shape == (1,)
    assert epistemic.shape == (1,)
    assert aleatoric.shape == (1,)
    assert epistemic.item() >= 0.0
    assert aleatoric.item() >= 0.0


@pytest.mark.skipif(not all_gnn_deps, reason="torch+pyg+rdkit+scatter not installed")
def test_train_fusion_smoke(tmp_path):
    """End-to-end smoke test: 2 epochs on synthetic complexes."""
    from caddack.gnn.train import train_fusion_from_complexes
    complexes = _make_synthetic_complexes(n=6)
    metrics = train_fusion_from_complexes(
        complexes=complexes,
        outdir=str(tmp_path / "fusion_out"),
        hidden_channels=16,
        num_gine_layers=2,
        num_geo_interactions=2,
        num_rbf=8,
        cutoff=5.0,
        bayesian_hidden=[16],
        epochs=2,
        batch_size=4,
        split="random",
        mc_samples_eval=3,
    )
    assert "mae" in metrics
    assert "r2" in metrics
    assert (tmp_path / "fusion_out" / "model.pt").exists()
    assert (tmp_path / "fusion_out" / "metrics.json").exists()
    assert (tmp_path / "fusion_out" / "config.json").exists()
