"""Tests for Vina's interaction terms.

These are a deliberate contrast to the learned towers: an explicit interaction
energy that cannot be computed from an atom inventory, only from the geometry.
Added after the affinity model was found to be reproducible by nine scalar
descriptors, i.e. to be ignoring the structure.
"""
import pytest

from caddack.docking.scoring import TERM_NAMES, WEIGHTS, vina_terms


def pair(z_a, z_b, distance):
    """One ligand atom and one pocket atom, a given centre distance apart."""
    return vina_terms([z_a, z_b], [(0.0, 0.0, 0.0), (distance, 0.0, 0.0)], n_ligand=1)


# --- the functional forms -----------------------------------------------------

def test_gauss1_peaks_at_touching_vdw_surfaces():
    """gauss1 is exp(-(d/0.5)^2) in surface distance, so it maxes at d = 0."""
    touching = 1.9 + 1.9                      # two carbon radii
    assert pair(6, 6, touching)["gauss1"] == pytest.approx(1.0, abs=1e-6)
    assert pair(6, 6, touching + 1.0)["gauss1"] < 0.02


def test_gauss2_peaks_three_angstrom_past_contact():
    """gauss2 is exp(-((d-3)/2)^2): a broad shoulder, not a contact term."""
    touching = 3.8
    at_peak = pair(6, 6, touching + 3.0)["gauss2"]
    assert at_peak == pytest.approx(1.0, abs=1e-6)
    assert pair(6, 6, touching)["gauss2"] < at_peak


def test_repulsion_is_the_squared_overlap_and_zero_otherwise():
    assert pair(6, 6, 3.8 - 1.3)["repulsion"] == pytest.approx(1.3 ** 2, abs=1e-6)
    assert pair(6, 6, 3.8)["repulsion"] == pytest.approx(0.0, abs=1e-9)
    assert pair(6, 6, 5.0)["repulsion"] == pytest.approx(0.0, abs=1e-9)


def test_hydrophobic_ramps_between_half_and_one_and_a_half_angstrom():
    touching = 3.8
    assert pair(6, 6, touching + 0.2)["hydrophobic"] == pytest.approx(1.0)
    assert pair(6, 6, touching + 2.0)["hydrophobic"] == pytest.approx(0.0)
    mid = pair(6, 6, touching + 1.0)["hydrophobic"]
    assert 0.0 < mid < 1.0


def test_hydrophobic_ignores_polar_pairs():
    """Only carbon-without-heteroatom-neighbours and halogens are hydrophobic."""
    assert pair(7, 8, 3.5)["hydrophobic"] == pytest.approx(0.0)
    assert pair(6, 6, 3.9)["hydrophobic"] > 0.0


def test_a_carbon_bonded_to_oxygen_is_not_hydrophobic():
    """Polarised carbon does not count, which is why bonds must be resolved."""
    # ligand: C bonded to O at 1.4 A. pocket: a lone carbon in contact.
    polarised = vina_terms([6, 8, 6], [(0, 0, 0), (1.4, 0, 0), (0, 3.9, 0)], n_ligand=2)
    bare = vina_terms([6, 6], [(0, 0, 0), (0, 3.9, 0)], n_ligand=1)
    assert bare["hydrophobic"] > polarised["hydrophobic"]


def test_hbond_needs_two_polar_atoms_and_overlap():
    n_o = vina_terms([7, 8], [(0, 0, 0), (2.9, 0, 0)], n_ligand=1)   # surface -0.6
    assert n_o["hbond"] > 0.0
    assert pair(6, 6, 2.9)["hbond"] == pytest.approx(0.0)
    assert vina_terms([7, 8], [(0, 0, 0), (3.6, 0, 0)], n_ligand=1)["hbond"] == \
        pytest.approx(0.0)


# --- bookkeeping --------------------------------------------------------------

def test_hydrogens_are_ignored():
    """Vina is united-atom. H also has no radius entry, so leaving it in would
    silently give it the fallback metal radius and distort every distance."""
    without = vina_terms([6, 6], [(0, 0, 0), (3.9, 0, 0)], n_ligand=1)
    with_h = vina_terms([6, 1, 6], [(0, 0, 0), (1.1, 0, 0), (3.9, 0, 0)], n_ligand=2)
    for key in TERM_NAMES:
        assert with_h[key] == pytest.approx(without[key], abs=1e-9)


def test_only_cross_interface_pairs_contribute():
    """Ligand-ligand and pocket-pocket pairs are intramolecular."""
    both_ligand = vina_terms([6, 6], [(0, 0, 0), (3.9, 0, 0)], n_ligand=2)
    assert both_ligand["gauss1"] == pytest.approx(0.0)
    assert both_ligand["n_contacts"] == 0


def test_weighted_sum_matches_the_published_weights():
    t = vina_terms([6, 7], [(0, 0, 0), (3.5, 0, 0)], n_ligand=1)
    assert t["vina_inter"] == pytest.approx(
        sum(WEIGHTS[k] * t[k] for k in TERM_NAMES), abs=1e-9)


def test_torsion_penalty_shrinks_the_score():
    """Vina divides by the torsion term, so it scales with the interaction."""
    rigid = vina_terms([6, 6], [(0, 0, 0), (3.9, 0, 0)], n_ligand=1, n_rotatable=0)
    floppy = vina_terms([6, 6], [(0, 0, 0), (3.9, 0, 0)], n_ligand=1, n_rotatable=10)
    assert rigid["vina_score"] < 0 and floppy["vina_score"] < 0
    assert floppy["vina_score"] > rigid["vina_score"]   # less favourable


def test_beyond_the_cutoff_nothing_contributes():
    assert pair(6, 6, 20.0)["n_contacts"] == 0
    assert pair(6, 6, 20.0)["vina_inter"] == pytest.approx(0.0)


def test_degenerate_inputs_do_not_raise():
    assert vina_terms([], [], n_ligand=0)["vina_inter"] == pytest.approx(0.0)
    assert vina_terms([6], [(0, 0, 0)], n_ligand=1)["vina_inter"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vina_terms([6, 6], [(0, 0, 0)], n_ligand=1)


def test_a_real_contact_is_favourable():
    """Sanity: carbon atoms at van der Waals contact should score negative."""
    assert vina_terms([6, 6], [(0, 0, 0), (3.9, 0, 0)], n_ligand=1)["vina_inter"] < 0


def test_module_imports_without_rdkit_or_vina():
    """Pure numpy: no optional dependency should be needed."""
    import caddack.docking.scoring as s

    assert s.vina_terms is not None
