"""Protein-ligand binding affinity prediction.

A random forest on 115 interaction features is the model that ships here. It was
not the first thing tried: a two-tower Bayesian graph network, with a GINE
encoder over the ligand and a SchNet-style encoder over the complex geometry,
came first and is still available in :mod:`caddack.gnn`. On a PDBbind scaffold
split the forest is ahead on every metric, and stayed ahead when the network was
given the same features, so the forest is what the package recommends.

    from caddack.affinity import features_from_files, load_model

    x = features_from_files("1a30_pocket.pdb", "1a30_ligand.mol2")
    model = load_model("models/affinity")
    pk, lo, hi = model.predict(x)
"""
from caddack.affinity.features import (
    DEFAULT_CUTOFF,
    FEATURE_DIM,
    feature_names,
    features_from_atoms,
    features_from_files,
)
from caddack.affinity.model import AffinityModel, load_model

__all__ = [
    "DEFAULT_CUTOFF",
    "FEATURE_DIM",
    "AffinityModel",
    "feature_names",
    "features_from_atoms",
    "features_from_files",
    "load_model",
]
