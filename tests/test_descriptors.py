"""Tests for QSAR featurisation and scaffold splitting.

``descriptors.py`` carries the most business logic in the package -- salt
stripping, canonicalisation, fingerprint generation across several RDKit API
branches -- and most of its failure modes are silent (a wrong answer, not an
exception), so the invariants are asserted directly.
"""
import importlib.util
import warnings

import pandas as pd
import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(not rdkit_available, reason="rdkit required")

# A spread of shapes: simple, aromatic, fused, drug-like, a salt, and a zwitterion.
SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "[Na+].CC(=O)[O-]",
    "c1ccc2ccccc2c1",
]


# --- canonicalisation ---------------------------------------------------------

@pytest.mark.parametrize("smi", SMILES)
def test_canonicalization_is_idempotent(smi):
    """Canonicalising an already-canonical SMILES must be a no-op."""
    from caddack.qsar.descriptors import canonicalize_smiles

    once = canonicalize_smiles(smi)
    assert once is not None
    assert canonicalize_smiles(once) == once


def test_canonicalization_unifies_equivalent_inputs():
    """Two spellings of one molecule must collapse to one canonical form."""
    from caddack.qsar.descriptors import canonicalize_smiles

    assert canonicalize_smiles("C1=CC=CC=C1") == canonicalize_smiles("c1ccccc1")


def test_invalid_smiles_returns_none_rather_than_raising():
    from caddack.qsar.descriptors import canonicalize_smiles, parse_smiles

    assert parse_smiles("not_a_smiles") is None
    assert canonicalize_smiles("not_a_smiles") is None
    assert parse_smiles(None) is None


# --- salt stripping -----------------------------------------------------------

def test_salt_stripping_keeps_the_largest_fragment():
    from caddack.qsar.descriptors import canonicalize_smiles, strip_salts

    kept = strip_salts("[Na+].CC(=O)[O-]")
    assert canonicalize_smiles(kept) == canonicalize_smiles("CC(=O)[O-]")


def test_salt_stripping_is_order_invariant():
    """Fragment order in the input must not change which fragment is kept."""
    from caddack.qsar.descriptors import strip_salts

    forward = "[Na+].CC(=O)[O-]"
    reversed_ = ".".join(reversed(forward.split(".")))
    assert strip_salts(forward) == strip_salts(reversed_)


def test_salt_stripping_passes_through_single_fragments():
    from caddack.qsar.descriptors import strip_salts

    assert strip_salts("CCO") == "CCO"


# --- fingerprints -------------------------------------------------------------

def test_ecfp_columns_are_named_by_diameter_not_radius():
    """The number in an ECFP name is the diameter: radius 2 is ECFP4."""
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert all(k.startswith("ECFP4_") for k in mol_to_ecfp_bits(mol, radius=2))
    assert all(k.startswith("ECFP2_") for k in mol_to_ecfp_bits(mol, radius=1))
    assert all(k.startswith("ECFP6_") for k in mol_to_ecfp_bits(mol, radius=3))


def test_fingerprint_is_deterministic_and_correctly_sized():
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    a = mol_to_ecfp_bits(mol, radius=2, n_bits=1024)
    b = mol_to_ecfp_bits(mol, radius=2, n_bits=1024)
    assert a == b                       # deterministic
    assert len(a) == 1024               # honours n_bits
    assert set(a.values()) <= {0, 1}    # genuinely a bit vector
    assert 0 < sum(a.values()) < 1024   # neither empty nor saturated


def test_identical_molecules_give_identical_fingerprints():
    """Two spellings of one molecule must produce the same bits."""
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles

    a = mol_to_ecfp_bits(parse_smiles("C1=CC=CC=C1"))
    b = mol_to_ecfp_bits(parse_smiles("c1ccccc1"))
    assert a == b


def test_different_molecules_give_different_fingerprints():
    from caddack.qsar.descriptors import mol_to_ecfp_bits, parse_smiles

    assert mol_to_ecfp_bits(parse_smiles("CCO")) != mol_to_ecfp_bits(parse_smiles("c1ccccc1"))


# --- dataframe featurisation --------------------------------------------------

def test_featurize_dataframe_flags_bad_rows_without_raising():
    from caddack.qsar.descriptors import featurize_dataframe

    df = pd.DataFrame({"SMILES": ["CCO", "not_a_smiles", "c1ccccc1"],
                       "pIC50": [5.0, 6.0, 7.0]})
    out = featurize_dataframe(df, smiles_col="SMILES")
    assert len(out) == 3
    assert (out["__error"] == "invalid_smiles").sum() == 1
    assert "SMILES_canonical" in out.columns


def test_featurize_dataframe_preserves_the_target():
    from caddack.qsar.descriptors import featurize_dataframe

    df = pd.DataFrame({"SMILES": ["CCO", "c1ccccc1"], "pIC50": [5.0, 7.0]})
    out = featurize_dataframe(df, smiles_col="SMILES", target_col="pIC50")
    assert out["pIC50"].tolist() == [5.0, 7.0]


# --- scaffold splitting -------------------------------------------------------

def test_acyclic_molecules_have_no_scaffold():
    """Ring-free inputs must not all collapse into one shared scaffold group."""
    from caddack.qsar.split import murcko_scaffold

    for smi in ("CCO", "CC(=O)O", "CCCCCC"):
        assert murcko_scaffold(smi) is None


def test_ring_molecules_have_a_scaffold():
    from caddack.qsar.split import murcko_scaffold

    assert murcko_scaffold("c1ccccc1CC") is not None
    assert murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") is not None


def test_scaffold_split_is_disjoint_by_scaffold():
    """The methodological claim, asserted: no scaffold spans both sides."""
    from caddack.qsar.split import murcko_scaffold, scaffold_split

    cores = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "c1ccc2ccccc2c1"]
    df = pd.DataFrame({"SMILES_canonical": [cores[i % len(cores)] + "C" * (i % 4)
                                            for i in range(60)]})
    train_idx, test_idx = scaffold_split(df, test_size=0.2, seed=0)

    train_scaf = {murcko_scaffold(s) for s in df.iloc[train_idx].SMILES_canonical}
    test_scaf = {murcko_scaffold(s) for s in df.iloc[test_idx].SMILES_canonical}
    assert train_scaf & test_scaf == set()
    assert len(train_idx) + len(test_idx) == len(df)


def test_scaffold_split_respects_requested_test_size():
    """A dominant scaffold group must not blow the test fraction wide open."""
    from caddack.qsar.split import scaffold_split

    cores = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "c1ccc2ccccc2c1",
             "c1cnc2ccccc2c1", "C1CCNCC1", "c1ccoc1"]
    df = pd.DataFrame({"SMILES_canonical": [cores[i % len(cores)] + "C" * (i % 5)
                                            for i in range(80)]})
    _, test_idx = scaffold_split(df, test_size=0.25, seed=0)
    achieved = len(test_idx) / len(df)
    assert 0.15 <= achieved <= 0.35, f"achieved {achieved:.2f}, wildly off 0.25"


def test_scaffold_split_warns_when_target_unreachable():
    """One dominant scaffold makes the target impossible; that must be loud."""
    from caddack.qsar.split import scaffold_split

    df = pd.DataFrame({"SMILES_canonical":
                       ["c1ccccc1" + "C" * i for i in range(90)]
                       + ["c1ccncc1" + "C" * i for i in range(10)]})
    with pytest.warns(RuntimeWarning, match="achieved test fraction"):
        scaffold_split(df, test_size=0.2, seed=1)


def test_scaffold_split_returns_positional_indices_on_filtered_frame():
    """Indices must address row positions even when the frame index has gaps."""
    from caddack.qsar.split import scaffold_split

    cores = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1"]
    df = pd.DataFrame({"SMILES_canonical": [cores[i % 4] + "C" * (i % 3)
                                            for i in range(40)]})
    filtered = df[df.SMILES_canonical.str.len() > 8]      # non-contiguous index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train_idx, test_idx = scaffold_split(filtered, test_size=0.25, seed=0)

    assert max(test_idx.tolist() + train_idx.tolist()) < len(filtered)
    # .iloc must select real rows, and the two sides must not overlap
    assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()


def test_acyclic_compounds_are_routed_to_train():
    """No-scaffold compounds must never land in the evaluation set."""
    from caddack.qsar.split import scaffold_split

    df = pd.DataFrame({"SMILES_canonical":
                       ["CCO", "CCC", "CCCC"]                       # acyclic
                       + [f"c1ccccc1{'C' * (i % 4)}" for i in range(12)]
                       + [f"c1ccncc1{'C' * (i % 4)}" for i in range(12)]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train_idx, test_idx = scaffold_split(df, test_size=0.2, seed=0)

    acyclic_positions = {0, 1, 2}
    assert acyclic_positions <= set(train_idx.tolist())
    assert acyclic_positions & set(test_idx.tolist()) == set()


def test_scaffold_split_report_records_what_happened():
    from caddack.qsar.split import scaffold_split

    df = pd.DataFrame({"SMILES_canonical":
                       ["c1ccccc1" + "C" * i for i in range(90)]
                       + ["c1ccncc1" + "C" * i for i in range(10)]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, report = scaffold_split(df, test_size=0.2, seed=1, return_report=True)

    d = report.as_dict()
    assert d["n_scaffold_groups"] == 2
    assert d["largest_group"] == 90
    assert d["requested_test_size"] == 0.2
    assert d["notes"], "an unreachable target must be recorded, not just warned"
