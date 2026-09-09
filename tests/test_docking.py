"""Tests for the docking module.

The heavy dependencies (RDKit, Meeko, Vina) are optional, so each test skips
cleanly when they are absent. Box maths and PDB cleaning need none of them.
"""
import importlib.util
import textwrap

import pytest

rdkit_available = importlib.util.find_spec("rdkit") is not None
meeko_available = importlib.util.find_spec("meeko") is not None
vina_available = importlib.util.find_spec("vina") is not None


# --- imports stay light -------------------------------------------------------

def test_module_imports_without_optional_deps():
    """caddack.docking must import even with no RDKit/Meeko/Vina installed."""
    import caddack.docking as d

    assert "dock_smiles" in d.__all__
    assert "box_from_reference_ligand" in d.__all__


# --- search box ---------------------------------------------------------------

def test_box_from_coords_centres_and_pads():
    from caddack.docking import box_from_coords

    coords = [(0.0, 0.0, 0.0), (10.0, 4.0, 2.0)]
    box = box_from_coords(coords, padding=3.0, min_size=1.0)
    assert box.center == pytest.approx((5.0, 2.0, 1.0))
    # extent + 2*padding on the spanning axis
    assert box.size[0] == pytest.approx(16.0)


def test_box_respects_minimum_size():
    """A tiny ligand must still get a box big enough to search in."""
    from caddack.docking import box_from_coords

    box = box_from_coords([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                          padding=1.0, min_size=16.0)
    assert min(box.size) == pytest.approx(16.0)


def test_box_rejects_empty_coords():
    from caddack.docking import box_from_coords

    with pytest.raises(ValueError):
        box_from_coords([])


# --- receptor cleaning --------------------------------------------------------

_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1      12.000  14.000  10.500  1.00  0.00           C
    ATOM      3  CA BALA A   1      12.500  14.500  10.900  0.50  0.00           C
    HETATM    4  O   HOH A 100      20.000  20.000  20.000  1.00  0.00           O
    HETATM    5 ZN    ZN A 200      15.000  15.000  15.000  1.00  0.00          ZN
    END
    """)


def _write(tmp_path, text, name="in.pdb"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_clean_receptor_drops_water_and_altloc(tmp_path):
    from caddack.docking import clean_receptor_pdb

    out = clean_receptor_pdb(_write(tmp_path, _PDB), tmp_path / "out.pdb")
    body = out.read_text()
    assert "HOH" not in body          # water removed
    assert "ZN" not in body           # HETATM dropped unless requested
    assert " BALA" not in body        # alternate location B dropped
    assert body.count("ATOM  ") == 2  # N and the altLoc-A CA survive


def test_clean_receptor_can_keep_cofactors(tmp_path):
    from caddack.docking import clean_receptor_pdb

    out = clean_receptor_pdb(_write(tmp_path, _PDB), tmp_path / "out.pdb",
                             keep_hetatm=["ZN"])
    body = out.read_text()
    assert "ZN" in body     # explicitly requested cofactor retained
    assert "HOH" not in body


def test_clean_receptor_rejects_empty(tmp_path):
    from caddack.docking import clean_receptor_pdb

    with pytest.raises(ValueError):
        clean_receptor_pdb(_write(tmp_path, "HEADER only\nEND\n"),
                           tmp_path / "out.pdb")


# --- ligand preparation -------------------------------------------------------

@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_embed_3d_produces_a_conformer():
    from caddack.docking import embed_3d

    mol = embed_3d("CCO")
    assert mol.GetNumConformers() == 1
    assert mol.GetNumAtoms() > 3  # hydrogens added


@pytest.mark.skipif(not rdkit_available, reason="rdkit required")
def test_embed_3d_rejects_bad_smiles():
    from caddack.docking import embed_3d

    with pytest.raises(ValueError):
        embed_3d("not_a_smiles")


@pytest.mark.skipif(not (rdkit_available and meeko_available),
                    reason="rdkit+meeko required")
def test_ligand_pdbqt_is_written(tmp_path):
    from caddack.docking import ligand_pdbqt_from_smiles

    out = ligand_pdbqt_from_smiles("CCO", tmp_path / "lig.pdbqt")
    text = out.read_text()
    assert "ATOM" in text
    assert "ROOT" in text  # Vina torsion tree marker


# --- docking ------------------------------------------------------------------

@pytest.mark.skipif(not (rdkit_available and meeko_available and vina_available),
                    reason="rdkit+meeko+vina required")
def test_dock_into_a_small_box_returns_ranked_poses(tmp_path):
    """End-to-end: prepare both sides, dock, and get sensibly ranked poses."""
    from caddack.docking import Box, dock_pdbqt, ligand_pdbqt_from_smiles, receptor_pdbqt_from_pdb

    # a small artificial receptor: a plane of alanines around the origin
    lines = []
    n = 0
    for i in range(-6, 7, 3):
        for j in range(-6, 7, 3):
            n += 1
            lines.append(
                f"ATOM  {n:5d}  CA  ALA A{n:4d}    {i:8.3f}{j:8.3f}"
                f"{-8.0:8.3f}  1.00  0.00           C"
            )
    receptor_pdb = tmp_path / "rec.pdb"
    receptor_pdb.write_text("\n".join(lines) + "\nEND\n")

    rec = receptor_pdbqt_from_pdb(receptor_pdb, tmp_path / "rec.pdbqt")
    lig = ligand_pdbqt_from_smiles("CCO", tmp_path / "lig.pdbqt")
    box = Box(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0))

    poses = dock_pdbqt(rec, lig, box, exhaustiveness=2, n_poses=3)
    assert 1 <= len(poses) <= 3
    assert all(p.coords for p in poses)
    # Vina returns poses ordered best (most negative) first
    scores = [p.score for p in poses]
    assert scores == sorted(scores)


@pytest.mark.skipif(not (rdkit_available and meeko_available and vina_available),
                    reason="rdkit+meeko+vina required")
def test_pose_rmsd_against_itself_is_zero(tmp_path):
    """RMSD of a pose against the molecule it came from must be ~0."""
    from caddack.docking import (
        Box,
        dock_pdbqt,
        ligand_pdbqt_from_smiles,
        pose_rmsd,
        pose_to_mol,
        receptor_pdbqt_from_pdb,
    )

    receptor_pdb = tmp_path / "rec.pdb"
    receptor_pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000  -8.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       3.000   0.000  -8.000  1.00  0.00           C\nEND\n"
    )
    rec = receptor_pdbqt_from_pdb(receptor_pdb, tmp_path / "rec.pdbqt")
    lig = ligand_pdbqt_from_smiles("CCO", tmp_path / "lig.pdbqt")
    poses = dock_pdbqt(rec, lig, Box((0.0, 0.0, 0.0), (18.0, 18.0, 18.0)),
                       exhaustiveness=2, n_poses=2)

    mol = pose_to_mol(poses[0])
    assert pose_rmsd(poses[0], mol) == pytest.approx(0.0, abs=1e-6)
