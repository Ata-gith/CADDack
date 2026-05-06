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


# --- qsar.split ---

def test_split_importable():
    from caddack.qsar.split import scaffold_split, murcko_scaffold  # noqa: F401


@pytest.mark.skipif(rdkit_available, reason="RDKit is installed; absence test not applicable")
def test_split_requires_rdkit_at_call_time():
    from caddack.qsar.split import murcko_scaffold

    with pytest.raises(ImportError, match="RDKit is required"):
        murcko_scaffold("CCO")


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_murcko_scaffold_returns_none_for_invalid():
    from caddack.qsar.split import murcko_scaffold

    assert murcko_scaffold("not_a_smiles!!!") is None
    assert murcko_scaffold("") is None
    assert murcko_scaffold("CCO") is not None  # valid, acyclic → empty scaffold ""


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_scaffold_split_sizes():
    import pandas as pd
    from caddack.qsar.split import scaffold_split

    # 10 simple molecules with varied scaffolds
    smiles = [
        "c1ccccc1",         # benzene
        "c1ccccc1C",        # toluene
        "c1ccccc1CC",       # ethylbenzene
        "c1ccncc1",         # pyridine
        "c1ccncc1C",        # methylpyridine
        "CCO", "CCCO", "CCCCO", "CC(=O)O", "CCC",
    ]
    df = pd.DataFrame({"SMILES_canonical": smiles})
    train, test = scaffold_split(df, smiles_col="SMILES_canonical", test_size=0.2, seed=42)

    assert len(train) + len(test) == len(df)
    assert set(train).isdisjoint(set(test))


@pytest.mark.skipif(not rdkit_available, reason="RDKit not installed")
def test_scaffold_split_invalid_smiles_go_to_train():
    import pandas as pd
    from caddack.qsar.split import scaffold_split

    smiles = ["CCO", "CCCO", "not_valid", "c1ccccc1", "c1ccncc1",
              "CCN", "CCC", "CCCC", "c1ccc(C)cc1", "c1ccoc1"]
    df = pd.DataFrame({"SMILES_canonical": smiles})
    invalid_idx = smiles.index("not_valid")

    train, test = scaffold_split(df, smiles_col="SMILES_canonical", test_size=0.2, seed=42)

    # invalid row must not appear in test set
    assert invalid_idx not in test
