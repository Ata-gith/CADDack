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
import re
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


# --- applicability-domain signal ----------------------------------------------

def test_ad_block_is_present_and_sane(data):
    ad = data.get("ad")
    assert ad, "the page's signal comparison needs the applicability-domain block"
    assert 0.0 < ad["threshold"] < 1.0
    assert ad["n_train"] > 0
    assert 0 <= ad["n_outside"] <= data["metrics"]["n_test"]


def test_every_complex_carries_a_similarity(data):
    sims = [r.get("ad_sim") for r in data["complexes"]]
    assert all(s is None or 0.0 <= s <= 1.0 for s in sims)
    assert sum(s is not None for s in sims) > 0.9 * len(sims)


def test_page_and_exporter_agree_on_who_is_out_of_domain(data):
    """Threshold and similarities must share a precision.

    Storing the threshold at full precision while rounding the per-complex
    similarities puts complexes within half a thousandth of the boundary on
    different sides depending on which one does the comparison, so the page
    reported a count the generating script disagreed with.
    """
    t = data["ad"]["threshold"]
    recomputed = sum(1 for r in data["complexes"]
                     if r.get("ad_sim") is not None and r["ad_sim"] < t)
    assert recomputed == data["ad"]["n_outside"]


def test_out_of_domain_ligands_really_are_harder(data):
    """The page claims they carry more error; check it against the records."""
    t = data["ad"]["threshold"]
    inside = [abs(r["pred"] - r["measured"]) for r in data["complexes"]
              if r.get("ad_sim") is not None and r["ad_sim"] >= t]
    outside = [abs(r["pred"] - r["measured"]) for r in data["complexes"]
               if r.get("ad_sim") is not None and r["ad_sim"] < t]
    assert inside and outside
    assert sum(outside) / len(outside) > sum(inside) / len(inside)


def test_uncertainty_panel_describes_the_shipped_model():
    """The forest is what the page ships, so the panel is about the forest. It
    used to chart five of the network's signals against each other, four of them
    indistinguishable from noise. The two it draws now must be computed from the
    bundles rather than written in by hand."""
    page = (BUNDLE.parent / "fusion.js").read_text()
    body = page[page.index("function drawForestUncertainty"):]
    body = body[:body.index("\n}")]

    assert "F.complexes" in body, "the panel must read the forest's predictions"
    assert "f.spread" in body, "the tree spread is the forest's own signal"
    assert "ad_sim" in body, "the domain distance must come from the bundle"
    assert "spearman" in body, "rho is quoted, so it must be computed"
    assert "rankAverage" not in page, "the dropped five-signal chart should stay dropped"
    assert "r.epi" not in body and "r.ale" not in body, (
        "the network's epistemic and aleatoric terms are no longer charted")


def test_tree_spread_outranks_the_model_free_lookup():
    """The forest's own uncertainty beats the distance lookup, which is the
    reverse of what the network managed and the reason the panel leads with it.
    Both bundles come from the same scaffold split, so joining them by pdb id is
    legitimate; if that ever stops being true this test fails on the join."""
    forest = json.loads((BUNDLE.parent / "affinity.json").read_text())
    network = json.loads(BUNDLE.read_text())
    net_by = {c["id"]: c for c in network["complexes"]}

    paired = [(f, net_by[f["id"]]) for f in forest["complexes"]
              if f["id"] in net_by and net_by[f["id"]].get("ad_sim") is not None]
    assert len(paired) == len(forest["complexes"]), (
        "the two bundles have drifted onto different splits; the panel's join is "
        "no longer sound")

    err = [abs(f["pred"] - f["measured"]) for f, _ in paired]
    spread = [f["spread"] for f, _ in paired]
    dist = [1 - n["ad_sim"] for _, n in paired]

    def rho(xs, ys):
        def rank(a):
            order = sorted(range(len(a)), key=lambda i: a[i])
            out = [0.0] * len(a)
            for pos, i in enumerate(order):
                out[i] = pos
            return out
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        vx = math.sqrt(sum((v - mx) ** 2 for v in rx))
        vy = math.sqrt(sum((v - my) ** 2 for v in ry))
        return cov / (vx * vy)

    spread_rho, dist_rho = rho(spread, err), rho(dist, err)
    assert spread_rho > 0.15, f"tree spread ranks errors at only {spread_rho:.3f}"
    assert dist_rho > 0, f"the distance lookup should point the right way, got {dist_rho:.3f}"
    assert spread_rho > dist_rho, (
        f"tree spread {spread_rho:.3f} no longer beats the lookup {dist_rho:.3f}; "
        "the panel's ordering would be wrong")


def test_viewer_draws_the_pocket_and_frames_it():
    """The pocket was styled with 3Dmol's `line`, which renders bonds only and at
    one pixel wide. Pocket atoms whose neighbours fell outside the 6 A cut had no
    bond left to draw and vanished; the rest were hairlines. The viewer also
    zoomed to the ligand alone, pushing what did render off-screen. Both made the
    panel look like it showed the ligand by itself."""
    page = (BUNDLE.parent / "fusion.js").read_text()
    body = page[page.index("function showStructure"):]
    body = body[:body.index("\n}")]

    assert "line:" not in body, (
        "`line` draws bonds only, so pocket atoms cut off from their neighbours "
        "render as nothing")
    # Every atom gets a sphere, so an atom with no bond is still drawn.
    assert body.count("sphere:") >= 2, (
        "both the pocket and the ligand need a per-atom sphere")
    assert "zoomTo()" in body, "the whole complex must be framed, not just the ligand"
    assert "zoomTo(ligand)" not in body


def test_viewer_pocket_colour_is_actually_grey():
    """The caption reads "grey = pocket". greyCarbon greys only the carbons and
    leaves N blue and O red, so the pocket read as stray coloured specks."""
    page = (BUNDLE.parent / "fusion.js").read_text()
    html = (BUNDLE.parent / "fusion.html").read_text()
    assert "grey = pocket" in html
    body = page[page.index("function showStructure"):]
    body = body[:body.index("\n}")]
    # Strip comments: the code explains why greyCarbon was wrong, and that
    # explanation must not trip a check aimed at the style it describes.
    code = re.sub(r"//.*", "", body)
    assert "POCKET_GREY" in code
    assert 'colorscheme: "greyCarbon"' not in code
    match = re.search(r'POCKET_GREY\s*=\s*"#([0-9a-fA-F]{6})"', page)
    assert match, "POCKET_GREY must be a literal hex colour"
    r, g, b = (int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    assert max(r, g, b) - min(r, g, b) <= 24, f"#{match.group(1)} is not grey"
    # Must separate from both panel backgrounds: #fff light, #1c1c22 dark.
    assert 90 < (r + g + b) / 3 < 190, (
        f"#{match.group(1)} will not read against both the light and dark panel")
