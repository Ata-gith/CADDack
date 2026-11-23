"""Tests for the affinity feature set and the shipped model wrapper.

A random forest on these 115 features reaches r 0.693 on a PDBbind scaffold
split, against 0.644 for the two-tower graph network it replaced, and stayed
ahead when the network was handed the same features.
"""
import importlib.util

import numpy as np
import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None


def test_feature_names_match_the_declared_dimension():
    from caddack.affinity import FEATURE_DIM, feature_names

    names = feature_names()
    assert len(names) == FEATURE_DIM == 113
    assert len(set(names)) == len(names), "duplicate feature name"


def test_feature_order_is_descriptors_then_size_then_physics():
    """The order is a contract: the browser builds the same vector by index."""
    from caddack.affinity import feature_names

    names = feature_names()
    assert names[:6] == ["MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds"]
    assert names[6] == "n_ligand_atoms"
    assert names[7:12] == ["gauss1", "gauss2", "repulsion", "hydrophobic", "hbond"]
    assert names[12] == "vina_inter"
    assert all(n.startswith("contact_") for n in names[13:])


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_features_from_atoms_builds_a_full_vector():
    from caddack.affinity import FEATURE_DIM, features_from_atoms

    # Ethanol as the ligand, a few carbons standing in for a pocket.
    z = [6, 6, 8] + [6] * 6
    pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.2, 1.2, 0.0)]
    pos += [(x, 4.0, 0.0) for x in range(6)]
    x = features_from_atoms(z, pos, n_ligand=3, smiles="CCO")
    assert x is not None
    assert x.shape == (FEATURE_DIM,)
    assert np.isfinite(x).all()


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_unparseable_smiles_returns_none_rather_than_a_partial_vector():
    from caddack.affinity import features_from_atoms

    z = [6, 6, 6]
    pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 4.0, 0.0)]
    assert features_from_atoms(z, pos, n_ligand=2, smiles="not_a_smiles") is None
    assert features_from_atoms(z, pos, n_ligand=2, smiles="") is None


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_pocket_size_is_not_a_feature():
    """It was, and came out second most important of 115 while being redundant and
    contaminated: PDBbind deposits hydrogens inconsistently, which inflated the
    count by about 24% at random. Removing it moved r by +0.003."""
    from caddack.affinity import feature_names

    names = feature_names()
    assert "n_pocket_atoms" not in names
    assert "pocket_per_ligand_atom" not in names
    assert "n_ligand_atoms" in names


def test_a_pocket_shorter_than_the_feature_radius_warns():
    """The interaction features reach 8 A. A 6 A pocket empties the outer shells
    and silently mismatches how the model was trained."""
    from caddack.affinity import features_from_atoms

    z = [6, 6, 8] + [6] * 6
    close = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.2, 1.2, 0.0)]
    close += [(float(x), 4.0, 0.0) for x in range(6)]        # all within ~4 A
    with pytest.warns(RuntimeWarning, match="pocket reaches only"):
        features_from_atoms(z, close, n_ligand=3, smiles="CCO")


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_a_distant_pocket_leaves_the_interaction_features_empty():
    """Interaction terms must come from the geometry, not from atom counts."""
    from caddack.affinity import feature_names, features_from_atoms

    names = feature_names()
    interaction = [i for i, n in enumerate(names)
                   if n.startswith("contact_") or n in ("gauss1", "gauss2",
                                                        "hydrophobic", "hbond")]
    z = [6, 6, 8] + [6] * 6
    far = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.2, 1.2, 0.0)]
    far += [(x, 500.0, 0.0) for x in range(6)]
    x = features_from_atoms(z, far, n_ligand=3, smiles="CCO")
    assert np.allclose(x[interaction], 0.0)


def test_model_wrapper_rejects_a_wrong_width_vector():
    from caddack.affinity import AffinityModel

    model = AffinityModel(forest=None, feature_names=["a", "b"], metrics={})
    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict(np.zeros(5))


def test_load_model_reports_a_missing_directory():
    from caddack.affinity import load_model

    with pytest.raises((FileNotFoundError, OSError)):
        load_model("/nonexistent/affinity")
