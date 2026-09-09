"""Tests for structure standardisation, applicability domain and packed bits.

These cover the cheminformatics controls: two spellings of one compound must
collapse to one record (otherwise duplicates split across train and test), and a
prediction outside the training chemistry must be flagged rather than returned
with false confidence.
"""
import importlib.util

import numpy as np
import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None


# --- standardisation ----------------------------------------------------------

@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
class TestStandardize:
    def test_salt_is_removed(self):
        from caddack.qsar.standardize import standardize_smiles

        assert standardize_smiles("Cl.CC(=O)Nc1ccc(O)cc1") == \
            standardize_smiles("CC(=O)Nc1ccc(O)cc1")

    def test_counterion_charge_is_neutralised(self):
        """A stripped hydrochloride keeps [NH3+] without neutralisation, which
        changes both descriptors and fingerprint bits."""
        from caddack.qsar.standardize import standardize_smiles

        assert standardize_smiles("CC(=O)[O-].[Na+]") == standardize_smiles("CC(=O)O")

    def test_neutralisation_can_be_disabled(self):
        from caddack.qsar.standardize import standardize_smiles

        charged = standardize_smiles("CC(=O)[O-].[Na+]", neutralize=False)
        assert charged is not None and "-" in charged

    def test_tautomers_collapse_when_requested(self):
        """Two tautomers of one compound must not survive as separate records."""
        from caddack.qsar.standardize import standardize_smiles

        a = standardize_smiles("O=C1CCCCC1", canonical_tautomer=True)
        b = standardize_smiles("OC1=CCCCC1", canonical_tautomer=True)
        assert a == b

    def test_invalid_input_returns_none(self):
        from caddack.qsar.standardize import standardize_smiles

        assert standardize_smiles("not_a_smiles!!!") is None
        assert standardize_smiles("") is None
        assert standardize_smiles(None) is None

    def test_inchikey_deduplicates_across_spellings(self):
        from caddack.qsar.standardize import inchikey

        assert inchikey("Cl.CC(=O)Nc1ccc(O)cc1") == inchikey("CC(=O)Nc1ccc(O)cc1")
        assert inchikey("C1=CC=CC=C1") == inchikey("c1ccccc1")

    def test_standardisation_is_idempotent(self):
        from caddack.qsar.standardize import standardize_smiles

        once = standardize_smiles("Cl.CC(=O)Nc1ccc(O)cc1")
        assert standardize_smiles(once) == once


# --- packed fingerprints ------------------------------------------------------

@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
class TestPackedFingerprints:
    def test_packed_round_trips_to_the_dense_dict(self):
        from caddack.qsar.descriptors import mol_to_ecfp_array, mol_to_ecfp_bits, parse_smiles

        mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
        packed = mol_to_ecfp_array(mol, radius=2, n_bits=2048)
        dense = mol_to_ecfp_bits(mol, radius=2, n_bits=2048)
        unpacked = np.unpackbits(packed)
        assert all(int(unpacked[i]) == dense[f"ECFP4_{i}"] for i in range(2048))

    def test_packed_is_64x_smaller(self):
        from caddack.qsar.descriptors import mol_to_ecfp_array, mol_to_ecfp_bits, parse_smiles

        mol = parse_smiles("CCO")
        packed = mol_to_ecfp_array(mol, n_bits=2048)
        dense_bytes = len(mol_to_ecfp_bits(mol, n_bits=2048)) * 8   # int64 per bit
        assert packed.nbytes == 256
        assert dense_bytes / packed.nbytes == 64


# --- parallel featurisation ---------------------------------------------------

@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_parallel_featurisation_matches_serial():
    """n_jobs must change only the speed, never the answer."""
    import pandas as pd

    from caddack.qsar.descriptors import featurize_dataframe

    cores = ["c1ccccc1", "c1ccncc1", "CC(=O)Oc1ccccc1C(=O)O"]
    df = pd.DataFrame({"SMILES": [cores[i % 3] + "C" * (i % 3) for i in range(24)]})
    serial = featurize_dataframe(df, smiles_col="SMILES", target_col=None)
    parallel = featurize_dataframe(df, smiles_col="SMILES", target_col=None, n_jobs=2)
    assert serial.equals(parallel)


# --- applicability domain -----------------------------------------------------

class TestSimilarityAD:
    @staticmethod
    def _fps(seed=0, n=40, n_bits=128, density=0.1):
        rng = np.random.default_rng(seed)
        return (rng.random((n, n_bits)) < density).astype(np.uint8)

    def test_tanimoto_of_identical_vectors_is_one(self):
        from caddack.qsar.applicability import tanimoto_matrix

        a = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        assert tanimoto_matrix(a, a)[0, 0] == pytest.approx(1.0)

    def test_tanimoto_of_disjoint_vectors_is_zero(self):
        from caddack.qsar.applicability import tanimoto_matrix

        a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        assert tanimoto_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_tanimoto_matches_the_definition(self):
        from caddack.qsar.applicability import tanimoto_matrix

        a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        # intersection 1, union 3
        assert tanimoto_matrix(a, b)[0, 0] == pytest.approx(1 / 3)

    def test_training_compounds_are_in_domain(self):
        from caddack.qsar.applicability import SimilarityAD

        train = self._fps()
        ad = SimilarityAD.fit(train, percentile=5.0)
        # each training compound is identical to itself, so similarity is 1.0
        assert ad.in_domain(train).all()

    def test_dissimilar_query_is_flagged_out_of_domain(self):
        from caddack.qsar.applicability import SimilarityAD

        train = self._fps(seed=0, density=0.3)
        ad = SimilarityAD.fit(train, percentile=50.0)
        # a query sharing no bits with anything in training
        far = np.zeros((1, train.shape[1]), dtype=np.uint8)
        far[0, :2] = 1
        assert ad.similarity(far)[0] < ad.threshold
        assert not ad.in_domain(far)[0]

    def test_fit_requires_enough_compounds(self):
        from caddack.qsar.applicability import SimilarityAD

        with pytest.raises(ValueError):
            SimilarityAD.fit(np.ones((1, 16), dtype=np.uint8))

    def test_is_serialisable_for_persisting_with_a_model(self):
        from caddack.qsar.applicability import SimilarityAD

        d = SimilarityAD.fit(self._fps()).as_dict()
        assert d["kind"] == "similarity_tanimoto"
        assert 0.0 <= d["threshold"] <= 1.0
        assert d["n_train"] == 40


class TestLeverageAD:
    def test_leverage_flags_an_extrapolated_point(self):
        from caddack.qsar.applicability import LeverageAD

        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        ad = LeverageAD.fit(X)
        assert ad.in_domain(X).mean() > 0.8          # most training points inside
        far = np.full((1, 5), 50.0)                   # far outside the cloud
        assert not ad.in_domain(far)[0]

    def test_h_star_follows_three_p_over_n(self):
        from caddack.qsar.applicability import LeverageAD

        rng = np.random.default_rng(0)
        ad = LeverageAD.fit(rng.normal(size=(100, 5)))
        assert ad.h_star == pytest.approx(3 * 5 / 100)

    def test_refuses_when_features_outnumber_compounds(self):
        """With 2048 fingerprint bits and few compounds the design matrix is
        singular, so leverage is meaningless -- it must refuse, not return junk."""
        from caddack.qsar.applicability import LeverageAD

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="more compounds than features"):
            LeverageAD.fit(rng.normal(size=(10, 50)))
