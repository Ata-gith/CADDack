import pytest


def test_gnn_dataset_requires_rdkit():
    from caddack.gnn.datasets import smiles_to_graph_arrays

    with pytest.raises(ImportError):
        smiles_to_graph_arrays("CCO")


def test_gnn_training_requires_torch():
    from caddack.gnn.train import train_from_csv

    with pytest.raises(ImportError):
        train_from_csv("missing.csv", smiles_col="SMILES", target_col="y")
