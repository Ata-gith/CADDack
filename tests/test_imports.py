import importlib.util

import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None
torch_available = importlib.util.find_spec("torch") is not None


# --- gnn.datasets ---

@pytest.mark.skipif(rdkit_available, reason="RDKit is installed; absence test not applicable")
def test_gnn_dataset_requires_rdkit():
    from caddack.gnn.datasets import smiles_to_graph_arrays

    with pytest.raises(ImportError):
        smiles_to_graph_arrays("CCO")


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_gnn_dataset_smiles_to_graph_arrays():
    from caddack.gnn.datasets import smiles_to_graph_arrays

    g = smiles_to_graph_arrays("CCO")
    assert len(g.node_features) == 3      # ethanol has 3 heavy atoms
    assert len(g.edge_index) == 4         # 2 bonds × 2 directed edges


# --- gnn.train ---

@pytest.mark.skipif(torch_available, reason="torch is installed; absence test not applicable")
def test_gnn_training_requires_torch():
    from caddack.gnn.train import train_from_csv

    with pytest.raises(ImportError):
        train_from_csv("missing.csv", smiles_col="SMILES", target_col="y")


# --- qsar.descriptors ---

def test_descriptors_importable():
    """Module must import cleanly regardless of whether RDKit is installed."""
    from caddack.qsar import (  # noqa: F401
        featurize_dataframe,
        smiles_to_features,
        parse_smiles,
        canonicalize_smiles,
        strip_salts,
    )


@pytest.mark.skipif(rdkit_available, reason="RDKit is installed; absence test not applicable")
def test_descriptors_requires_rdkit_at_call_time():
    from caddack.qsar.descriptors import parse_smiles

    with pytest.raises(ImportError, match="RDKit is required"):
        parse_smiles("CCO")


@pytest.mark.skipif(rdkit_available, reason="RDKit is installed; absence test not applicable")
def test_smiles_to_features_requires_rdkit():
    from caddack.qsar.descriptors import smiles_to_features

    with pytest.raises(ImportError, match="RDKit is required"):
        smiles_to_features("CCO")


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_parse_smiles_returns_mol():
    from caddack.qsar.descriptors import parse_smiles

    assert parse_smiles("CCO") is not None
    assert parse_smiles("not_a_smiles!!!") is None
    assert parse_smiles(123) is None  # type: ignore[arg-type]


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_smiles_to_features_returns_dict():
    from caddack.qsar.descriptors import smiles_to_features

    feats = smiles_to_features("CCO")
    assert feats is not None
    assert "MolWt" in feats
    assert "LogP" in feats
    assert "SMILES_canonical" in feats
    assert feats["SMILES_canonical"] == "CCO"


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_strip_salts_keeps_largest_fragment():
    from caddack.qsar.descriptors import strip_salts

    # sodium acetate: acetic acid is larger than Na
    assert strip_salts("CC(=O)[O-].[Na+]") == "CC(=O)[O-]"
    assert strip_salts("CCO") == "CCO"  # no salt, unchanged
