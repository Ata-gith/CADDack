"""Guard the contract between the Python model and the browser demo.

``docs/`` ships a model trained by ``scripts/build_demo_model.py`` and evaluated
in JavaScript. If the two disagree about how a molecule becomes a feature
vector, the page serves confident, wrong numbers with nothing to indicate it.

The failure that motivated these tests: RDKit.js accepts ``useChirality`` but
silently ignores ``includeChirality``, the spelling the Python API uses. Passing
the wrong one returns a different fingerprint with no error, which showed up
only as a Jaccard of 0.714 against Python on a chiral molecule.

These tests check the Python half of the contract and the shape of what is
shipped. Checking the JavaScript half needs a browser; ``docs/README.md``
records how that was verified.
"""
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None

MODEL = Path(__file__).resolve().parents[1] / "docs" / "model.json"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="docs/model.json not built")


@pytest.fixture(scope="module")
def bundle():
    return json.loads(MODEL.read_text())


def _decode(b64: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


# --- what the page relies on --------------------------------------------------

def test_bundle_has_everything_the_page_reads(bundle):
    for key in ("forest", "train", "ad", "feature_order", "n_bits", "metrics"):
        assert key in bundle, f"docs/app.js reads {key!r}"
    for key in ("n_trees", "offsets", "feature", "threshold", "left", "right", "value"):
        assert key in bundle["forest"]


def test_feature_order_matches_the_python_descriptor_list(bundle):
    """The browser builds [descriptors..., bits...] in this exact order."""
    from caddack.qsar.train_qsar import NUM_DESC

    assert bundle["feature_order"] == list(NUM_DESC)


def test_offsets_describe_the_flattened_forest(bundle):
    f = bundle["forest"]
    offsets = f["offsets"]
    assert len(offsets) == f["n_trees"] + 1
    assert offsets[0] == 0
    assert all(b > a for a, b in zip(offsets, offsets[1:]))
    # Every flat array must hold exactly offsets[-1] nodes.
    n = offsets[-1]
    assert len(_decode(f["feature"], np.int16)) == n
    assert len(_decode(f["threshold"], np.float32)) == n
    assert len(_decode(f["left"], np.int32)) == n
    assert len(_decode(f["right"], np.int32)) == n
    assert len(_decode(f["value"], np.float32)) == n


def test_tree_children_are_in_range(bundle):
    """A child index outside its tree would send the JS walker into another tree."""
    f = bundle["forest"]
    left = _decode(f["left"], np.int32)
    right = _decode(f["right"], np.int32)
    for t in range(f["n_trees"]):
        lo, hi = f["offsets"][t], f["offsets"][t + 1]
        size = hi - lo
        for arr in (left[lo:hi], right[lo:hi]):
            assert arr.min() >= -1
            assert arr.max() < size


def test_split_features_are_valid_column_indices(bundle):
    f = bundle["forest"]
    feature = _decode(f["feature"], np.int16)
    n_features = len(bundle["feature_order"]) + bundle["n_bits"]
    internal = feature[feature >= 0]
    assert internal.max() < n_features


def test_training_set_arrays_agree_on_length(bundle):
    tr = bundle["train"]
    n, n_bits = tr["n"], tr["n_bits"]
    assert len(tr["smiles"]) == n
    assert len(_decode(tr["logS"], np.float32)) == n
    assert len(base64.b64decode(tr["fps"])) == n * (n_bits // 8)


def test_metrics_are_from_a_scaffold_split(bundle):
    """The page claims a scaffold split; the bundle must back that up."""
    assert bundle["split"]["n_scaffold_groups"] > 1
    assert bundle["metrics"]["n_test"] >= 50
    assert 0.0 < bundle["ad"]["threshold"] < 1.0


def test_bundle_is_strict_json():
    """NaN is not valid JSON and would break the browser's fetch().json()."""
    raw = MODEL.read_text()
    for token in ("NaN", "Infinity"):
        assert token not in raw


# --- the Python half of the featurisation contract ----------------------------

@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_fingerprints_depend_on_chirality(bundle):
    """If this ever stops being true, the useChirality flag no longer matters.

    It is the whole reason the browser must pass useChirality: these two
    molecules differ only in stereochemistry, and the trained model can tell
    them apart.
    """
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles

    n_bits = bundle["n_bits"]
    r, s = parse_smiles("C[C@H](N)C(=O)O"), parse_smiles("C[C@@H](N)C(=O)O")
    assert mol_to_ecfp_bits(r, n_bits=n_bits) != mol_to_ecfp_bits(s, n_bits=n_bits)


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_packed_fingerprints_round_trip_to_the_dense_bits(bundle):
    """The browser unpacks these bytes for Tanimoto; packing must be MSB-first."""
    from caddack.qsar.descriptors import mol_to_ecfp_array, mol_to_ecfp_bits, parse_smiles

    n_bits = bundle["n_bits"]
    mol = parse_smiles(bundle["train"]["smiles"][0])
    unpacked = np.unpackbits(mol_to_ecfp_array(mol, n_bits=n_bits))
    dense = mol_to_ecfp_bits(mol, n_bits=n_bits)
    assert all(int(unpacked[i]) == dense[f"ECFP4_{i}"] for i in range(n_bits))


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_shipped_fingerprints_match_their_smiles(bundle):
    """The packed blob must correspond to the SMILES shipped beside it."""
    from caddack.qsar.descriptors import mol_to_ecfp_array, parse_smiles

    tr = bundle["train"]
    per_mol = tr["n_bits"] // 8
    raw = np.frombuffer(base64.b64decode(tr["fps"]), np.uint8)
    for i in (0, tr["n"] // 2, tr["n"] - 1):
        expected = mol_to_ecfp_array(parse_smiles(tr["smiles"][i]), n_bits=tr["n_bits"])
        assert np.array_equal(raw[i * per_mol:(i + 1) * per_mol], expected)


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_salt_stripping_changes_the_features():
    """The browser must strip salts too, or multi-fragment inputs diverge.

    This is the third load-bearing detail of the contract and the one that
    actually slipped through: featurising "[Na+].[Cl-]" whole rather than its
    largest fragment gave a different prediction in the browser than in Python,
    silently, for every salt and co-crystal.
    """
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles, strip_salts

    whole = "N.N.Cl[Pt]Cl"
    parent = strip_salts(whole)
    assert parent == "Cl[Pt]Cl"
    assert (mol_to_ecfp_bits(parse_smiles(whole), n_bits=256)
            != mol_to_ecfp_bits(parse_smiles(parent), n_bits=256))


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_salt_stripping_keeps_the_largest_fragment_and_breaks_ties_first():
    """Ties go to the first fragment: Python compares with strict `>`."""
    from caddack.qsar.descriptors import strip_salts

    assert strip_salts("CC(=O)[O-].[Na+]") == "CC(=O)[O-]"
    assert strip_salts("[Na+].[Cl-]") == "[Na+]"   # equal size, first wins
    assert strip_salts("CCO") == "CCO"


def test_demo_app_strips_salts():
    """docs/app.js must implement the stripping, not just document it."""
    app = (Path(__file__).resolve().parents[1] / "docs" / "app.js").read_text()
    assert "stripSalts" in app
    assert "const parent = stripSalts(smiles)" in app
