"""Guard the contract between the fusion exporter and the affinity demo page.

``docs/fusion.json`` is produced by ``scripts/build_fusion_demo.py`` and read by
``docs/fusion.js``. Unlike the solubility demo the model does not run in the
browser, so there is no featurisation to keep in step — but the page makes
claims about the data (a scaffold split, an uncertainty decomposition, a subset
with 3D coordinates) and those claims have to hold.

The failure worth guarding against: predictions and PDB identifiers falling out
of alignment. The benchmark script drops unparseable complexes silently, so the
exporter keeps surviving examples beside their tensors; if that ever regresses,
every structure on the page would be labelled with another complex's numbers and
nothing would look wrong.
"""
import json
import math
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "docs" / "fusion.json"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="docs/fusion.json not built")


@pytest.fixture(scope="module")
def data():
    return json.loads(BUNDLE.read_text())


def test_has_everything_the_page_reads(data):
    for key in ("metrics", "split", "complexes", "reliability", "structures"):
        assert key in data, f"docs/fusion.js reads {key!r}"
    for key in ("mae", "r2", "pearson", "coverage_1sigma", "coverage_2sigma",
                "n_train", "n_test", "baseline_mae"):
        assert key in data["metrics"]


def test_every_complex_record_is_complete(data):
    for r in data["complexes"]:
        for key in ("id", "measured", "pred", "epi", "ale", "n_ligand", "n_pocket"):
            assert key in r, f"{r.get('id')} missing {key}"
        assert math.isfinite(r["pred"]) and math.isfinite(r["measured"])
        # Uncertainties are standard deviations: non-negative by construction.
        assert r["epi"] >= 0 and r["ale"] >= 0
        assert r["n_ligand"] > 0 and r["n_pocket"] > 0


def test_complex_ids_are_unique(data):
    ids = [r["id"] for r in data["complexes"]]
    assert len(ids) == len(set(ids))


def test_record_count_matches_reported_n_test(data):
    assert len(data["complexes"]) == data["metrics"]["n_test"]


def test_affinities_are_in_plausible_pk_units(data):
    """PDBbind -log Kd/Ki spans roughly 0-16; anything outside means a unit slip."""
    for r in data["complexes"]:
        assert 0 <= r["measured"] <= 16, f"{r['id']} measured={r['measured']}"
        assert -5 <= r["pred"] <= 20, f"{r['id']} pred={r['pred']}"


def test_metrics_agree_with_the_shipped_records(data):
    """Recompute MAE from the records; a mismatch means they fell out of step."""
    errs = [abs(r["pred"] - r["measured"]) for r in data["complexes"]]
    assert sum(errs) / len(errs) == pytest.approx(data["metrics"]["mae"], abs=1e-3)


def test_it_beats_predicting_the_mean(data):
    """A model that cannot beat the training mean has learned nothing useful."""
    assert data["metrics"]["mae"] < data["metrics"]["baseline_mae"]


def test_split_was_by_scaffold(data):
    """The page says the test ligands are chemically novel; back that up."""
    assert data["split"]["n_scaffold_groups"] > 100
    assert data["split"]["achieved_test_size"] == pytest.approx(0.2, abs=0.05)


def test_coverage_is_reported_as_a_fraction(data):
    for key in ("coverage_1sigma", "coverage_2sigma"):
        assert 0.0 <= data["metrics"][key] <= 1.0
    assert data["metrics"]["coverage_2sigma"] >= data["metrics"]["coverage_1sigma"]


def test_reliability_bins_cover_the_test_set(data):
    """The honesty chart must describe every test complex, not a subset."""
    bins = data["reliability"]
    assert len(bins) >= 3
    assert sum(b["n"] for b in bins) == data["metrics"]["n_test"]
    # Bins are ordered by increasing claimed uncertainty.
    unc = [b["mean_uncertainty"] for b in bins]
    assert unc == sorted(unc)
    for b in bins:
        assert b["mean_abs_error"] >= 0


def test_structures_reference_real_complexes(data):
    """A structure keyed to an id with no record would render unlabelled."""
    ids = {r["id"] for r in data["complexes"]}
    assert data["structures"], "no structures shipped; the 3D viewer would be empty"
    for pid in data["structures"]:
        assert pid in ids, f"structure {pid} has no prediction record"


def test_structure_coordinates_are_well_formed(data):
    """Three floats per atom, ligand atoms first, matching the record's counts."""
    by_id = {r["id"]: r for r in data["complexes"]}
    for pid, s in data["structures"].items():
        n = len(s["elements"])
        assert len(s["xyz"]) == 3 * n, f"{pid}: xyz length does not match atom count"
        assert all(math.isfinite(v) for v in s["xyz"])
        assert 0 < s["n_ligand"] < n, f"{pid}: ligand must be a proper prefix"
        rec = by_id[pid]
        assert s["n_ligand"] == rec["n_ligand"]
        assert n == rec["n_ligand"] + rec["n_pocket"]


def test_structures_are_not_all_at_the_origin(data):
    """Guards against an export that writes zeros and renders a blank viewer."""
    for pid, s in data["structures"].items():
        assert any(abs(v) > 1e-6 for v in s["xyz"]), f"{pid}: all coordinates are zero"


def test_bundle_is_strict_json():
    """NaN is not valid JSON and would break the page's fetch().json()."""
    raw = BUNDLE.read_text()
    for token in ("NaN", "Infinity"):
        assert token not in raw


def test_page_states_that_predictions_are_precomputed():
    """The honesty claim is part of the deliverable, not just the commit message."""
    page = (BUNDLE.parent / "fusion.html").read_text()
    assert "computed ahead of time" in page
    assert "index.html" in page, "the two demo pages should link to each other"
