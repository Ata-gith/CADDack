import pytest


def test_gnn_dataset_requires_rdkit():
    from caddack.gnn.datasets import smiles_to_graph_arrays

    with pytest.raises(ImportError):
        smiles_to_graph_arrays("CCO")


def test_gnn_training_requires_torch():
    from caddack.gnn.train import train_from_csv

    with pytest.raises(ImportError):
        train_from_csv("missing.csv", smiles_col="SMILES", target_col="y")


# --- qsar.descriptors lazy-import tests ---

def test_descriptors_import_without_rdkit():
    """Module must be importable even when RDKit is absent."""
    from caddack.qsar import (  # noqa: F401
        featurize_dataframe,
        smiles_to_features,
        parse_smiles,
        canonicalize_smiles,
        strip_salts,
    )


def test_descriptors_requires_rdkit_at_call_time():
    from caddack.qsar.descriptors import parse_smiles

    with pytest.raises(ImportError, match="RDKit is required"):
        parse_smiles("CCO")


def test_smiles_to_features_requires_rdkit():
    from caddack.qsar.descriptors import smiles_to_features

    with pytest.raises(ImportError, match="RDKit is required"):
        smiles_to_features("CCO")
