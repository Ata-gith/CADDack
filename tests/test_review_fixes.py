"""Regression tests for the correctness fixes from the main-branch audit."""
import importlib.util

import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None
torch_available = importlib.util.find_spec("torch") is not None
pyg_available = importlib.util.find_spec("torch_geometric") is not None
gnn_deps = rdkit_available and torch_available and pyg_available


# --- geometry: element inference (no optional deps needed) ---

def test_element_inference_column_aware():
    """Blank element columns must resolve to the right element via PDB columns:
    ' CA ' -> carbon (not calcium), ' ND1' -> N, ' OD1' -> O, 'FE  ' -> iron."""
    import tempfile

    from caddack.gnn.geometry import parse_pdb_atoms

    pdb = (
        "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00  0.00\n"
        "ATOM      2  CB  ALA A   1      12.000  14.000  10.500  1.00  0.00\n"
        "ATOM      3  ND1 HIS A   2      13.000  15.000  11.000  1.00  0.00\n"
        "ATOM      4  OD1 ASP A   3      14.000  16.000  12.000  1.00  0.00\n"
        "ATOM      5  SD  MET A   4      15.000  17.000  13.000  1.00  0.00\n"
        "HETATM    6 FE   HEM A   5      16.000  18.000  14.000  1.00  0.00\n"
    )
    p = tempfile.mktemp(suffix=".pdb")
    with open(p, "w") as f:
        f.write(pdb)
    z = {a.name: a.atomic_num for a in parse_pdb_atoms(p)}
    assert z["CA"] == 6    # alpha-carbon, NOT calcium (20)
    assert z["CB"] == 6
    assert z["ND1"] == 7
    assert z["OD1"] == 8
    assert z["SD"] == 16
    assert z["FE"] == 26   # genuine two-letter element


def test_populated_element_column_wins():
    """When columns 77-78 carry the element, it is used verbatim."""
    import tempfile

    from caddack.gnn.geometry import parse_pdb_atoms

    line = "ATOM      1  CA  ALA A   1      11.104  13.207  10.000  1.00  0.00           C\n"
    p = tempfile.mktemp(suffix=".pdb")
    with open(p, "w") as f:
        f.write(line)
    atoms = parse_pdb_atoms(p)
    assert atoms[0].atomic_num == 6


# --- datasets: empty SMILES is treated as invalid ---

@pytest.mark.skipif(not (rdkit_available and torch_available), reason="rdkit+torch required")
def test_empty_smiles_skipped():
    from caddack.gnn.datasets import build_pyg_dataset

    with pytest.warns(UserWarning, match="skipped 1"):
        ds = build_pyg_dataset(["CCO", "", "c1ccccc1"], [1.0, 2.0, 3.0], skip_invalid=True)
    assert len(ds) == 2
    assert all(d.x.dim() == 2 and d.x.shape[1] == 7 for d in ds)


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_empty_smiles_strict_raises():
    from caddack.gnn.datasets import smiles_to_graph_arrays

    with pytest.raises(ValueError):
        smiles_to_graph_arrays("")


# --- train: classification with a continuous target is rejected ---

def test_classification_continuous_target_raises():
    import pandas as pd

    from caddack.gnn.train import _prepare_frame

    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "y": [5.2, 6.1]})
    with pytest.raises(ValueError, match="not binary"):
        _prepare_frame(df, "smiles", "y", task="classification", positive_threshold=None)


def test_classification_binary_target_ok():
    import pandas as pd

    from caddack.gnn.train import _prepare_frame

    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "y": [0, 1]})
    out = _prepare_frame(df, "smiles", "y", task="classification", positive_threshold=None)
    assert set(out["y"]) == {0, 1}


# --- models: predict_with_uncertainty is NaN-free for n_samples=1 ---

@pytest.mark.skipif(not gnn_deps, reason="torch+pyg+rdkit required")
def test_uncertainty_no_nan_single_sample():
    import torch
    from torch_geometric.data import Batch, Data

    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet

    m = FusionAffinityNet.build(ligand_in_channels=7, ligand_edge_dim=7, hidden_channels=8,
                                num_gine_layers=1, num_geo_interactions=1, num_rbf=4, cutoff=5.0,
                                bayesian_hidden=[8])
    lig = Batch.from_data_list([to_pyg_data(smiles_to_graph_arrays("CCO"), y=1.0)])
    geo = Batch.from_data_list([Data(z=torch.tensor([6, 7, 8]), pos=torch.randn(3, 3))])
    _, ep, al = m.predict_with_uncertainty(lig, geo, n_samples=1)
    assert torch.isfinite(ep).all() and torch.isfinite(al).all()


@pytest.mark.skipif(not gnn_deps, reason="torch+pyg+rdkit required")
def test_predict_restores_training_mode():
    import torch
    from torch_geometric.data import Batch, Data

    from caddack.gnn.datasets import smiles_to_graph_arrays, to_pyg_data
    from caddack.gnn.models import FusionAffinityNet

    m = FusionAffinityNet.build(ligand_in_channels=7, ligand_edge_dim=7, hidden_channels=8,
                                num_gine_layers=1, num_geo_interactions=1, num_rbf=4, cutoff=5.0,
                                bayesian_hidden=[8])
    lig = Batch.from_data_list([to_pyg_data(smiles_to_graph_arrays("CCO"), y=1.0)])
    geo = Batch.from_data_list([Data(z=torch.tensor([6, 7, 8]), pos=torch.randn(3, 3))])
    m.train()
    m.predict_with_uncertainty(lig, geo, n_samples=2)
    assert m.training is True  # mode restored after the eval() inside


# --- train: target standardisation removes the affinity offset ---

def test_standardize_target_config_roundtrip(tmp_path):
    """train_fusion_from_complexes records y_mean/y_std so predictions rescale."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("rdkit")
    import json

    from caddack.gnn.geometry import ComplexExample, GeometryRecord
    from caddack.gnn.train import train_fusion_from_complexes

    def mk(smiles, aff):
        n_lig, n_pocket = 6, 12
        geo = GeometryRecord(
            atomic_nums=[6] * (n_lig + n_pocket),
            positions=[(float(i) * 0.5, 0.0, 0.0) for i in range(n_lig + n_pocket)],
            n_ligand=n_lig, n_pocket=n_pocket)
        return ComplexExample(pdb_id="T", affinity=aff,
                              ligand_smiles=smiles, geo=geo)

    # affinities centred far from zero: standardisation must capture the offset
    smis = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCC", "CCOC", "c1ccncc1", "CCCl"]
    complexes = [mk(s, 6.0 + 0.5 * i) for i, s in enumerate(smis)]

    out = tmp_path / "fusion"
    train_fusion_from_complexes(
        complexes=complexes, outdir=str(out), hidden_channels=16,
        num_gine_layers=1, num_geo_interactions=1, num_rbf=8, cutoff=5.0,
        bayesian_hidden=[16], epochs=2, batch_size=4, test_size=0.25,
        split="random", mc_samples_eval=3)

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["standardize_target"] is True
    # y_mean should sit in the affinity range, not at 0
    assert 5.0 < cfg["y_mean"] < 11.0
    assert cfg["y_std"] > 0
