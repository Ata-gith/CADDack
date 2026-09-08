"""Receptor, ligand and search-box preparation for docking.

AutoDock Vina consumes PDBQT files (coordinates plus atom types and partial
charges) and a search box. This module turns the inputs CADDack already works
with — a SMILES string, a receptor PDB, a reference ligand — into those.

Every optional dependency is imported lazily, so the module imports cleanly
without RDKit, Meeko or Vina installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class Box:
    """Vina search box, in angstroms."""

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "RDKit is required to prepare ligands. "
            "Install with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem, AllChem


def _require_meeko():
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Meeko is required to write PDBQT files. Install with `pip install meeko`."
        ) from exc
    return MoleculePreparation, PDBQTWriterLegacy


# --------------------------------------------------------------------- ligand
def embed_3d(smiles: str, seed: int = 42, max_iterations: int = 0):
    """SMILES -> RDKit Mol with one embedded, force-field-relaxed conformer.

    Hydrogens are added before embedding (geometry depends on them) and kept,
    since Vina's atom typing needs to know which heavy atoms are polar-H donors.
    """
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = max_iterations  # 0 = RDKit picks a sensible default
    if AllChem.EmbedMolecule(mol, params) != 0:
        # ETKDG can fail on strained or unusual systems; random coords is the
        # documented fallback and still gives Vina a usable starting geometry.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"Could not embed a 3D conformer for {smiles!r}")

    # MMFF is preferred; UFF covers atom types MMFF does not parameterise.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # an unrelaxed conformer still docks; Vina re-optimises torsions

    return mol


def ligand_pdbqt_from_smiles(smiles: str, out_path: str | Path,
                             seed: int = 42) -> Path:
    """Write a docking-ready ligand PDBQT from SMILES."""
    mol = embed_3d(smiles, seed=seed)
    return ligand_pdbqt_from_mol(mol, out_path)


def ligand_pdbqt_from_mol(mol, out_path: str | Path) -> Path:
    """Write a docking-ready ligand PDBQT from an RDKit Mol with a conformer."""
    MoleculePreparation, PDBQTWriterLegacy = _require_meeko()
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D conformer; call embed_3d() first")

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    if not setups:
        raise ValueError("Meeko could not prepare this ligand")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"Meeko failed to write ligand PDBQT: {err}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pdbqt, encoding="utf-8")
    return out_path


# ------------------------------------------------------------------- receptor
# Solvent and common crystallisation additives. These sit in the site but are
# not part of the target, so leaving them in would block the search box.
_DISCARD_HETATM = {
    "HOH", "WAT", "DOD",              # water
    "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT", "DMS", "TRS", "IMD",
}


def clean_receptor_pdb(pdb_path: str | Path, out_path: str | Path,
                       keep_hetatm: Optional[Sequence[str]] = None) -> Path:
    """Strip waters/additives, drop alternate locations, keep the first model.

    Cofactors and metals that matter for binding can be retained by name via
    ``keep_hetatm`` (e.g. ``["ZN", "HEM"]``).
    """
    keep = {r.upper() for r in (keep_hetatm or [])}
    lines: List[str] = []
    text = Path(pdb_path).read_text(errors="ignore")

    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break  # first model only
        if record not in ("ATOM", "HETATM"):
            continue
        resname = line[17:20].strip().upper()
        # HETATM is dropped unless explicitly requested: this is what removes the
        # co-crystallised ligand, so the site is empty for re-docking.
        if record == "HETATM" and resname not in keep:
            continue
        alt_loc = line[16:17]
        if alt_loc not in (" ", "A", ""):
            continue  # keep one conformer per atom
        lines.append(line)

    if not lines:
        raise ValueError(f"No usable ATOM records in {pdb_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return out_path


def _default_his_templates(pdb_path: Path) -> dict:
    """Assign every histidine the neutral Nε-H tautomer (HIE).

    Crystal structures carry no hydrogens, so HIE/HID/HIP are indistinguishable
    from heavy atoms alone and Meeko refuses to guess. HIE is the conventional
    default at physiological pH. Override per residue if you know better —
    a protonated His coordinating a metal or catalytic site may need HIP.
    """
    templates = {}
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        if line[17:20].strip().upper() != "HIS":
            continue
        chain = line[21:22].strip()
        resseq = line[22:26].strip()
        templates[f"{chain}:{resseq}"] = "HIE"
    return templates


def receptor_pdbqt_from_pdb(pdb_path: str | Path, out_path: str | Path,
                            keep_hetatm: Optional[Sequence[str]] = None,
                            strict: bool = False) -> Path:
    """Clean a receptor PDB and convert it to PDBQT for Vina.

    Prefers Meeko's residue-template path, which adds polar hydrogens and gives
    proper AutoDock donor/acceptor typing. That path needs a well-formed polymer,
    so pass a **complete protein**, not a truncated pocket file.

    If it fails (interrupted residues, unknown monomers), this falls back to a
    minimal element-based writer unless ``strict=True``. The fallback types every
    N as an acceptor and carries no polar hydrogens, which weakens the hydrogen
    bonding term and measurably degrades pose accuracy — prefer fixing the input.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = out_path.with_suffix(".clean.pdb")
    clean_receptor_pdb(pdb_path, cleaned, keep_hetatm=keep_hetatm)

    try:
        from meeko import (MoleculePreparation, PDBQTWriterLegacy, Polymer,
                           ResidueChemTemplates)

        polymer = Polymer.from_pdb_string(
            cleaned.read_text(),
            chem_templates=ResidueChemTemplates.create_from_defaults(),
            mk_prep=MoleculePreparation(),
            set_template=_default_his_templates(cleaned),
            allow_bad_res=True,
        )
        written = PDBQTWriterLegacy.write_string_from_polymer(polymer)
        text = written[0] if isinstance(written, (tuple, list)) else written
        if text:
            out_path.write_text(str(text), encoding="utf-8")
            return out_path
        raise ValueError("Meeko returned an empty receptor PDBQT")
    except Exception as exc:
        if strict:
            raise ValueError(
                f"Meeko receptor preparation failed for {pdb_path}: {exc}"
            ) from exc
        import warnings
        warnings.warn(
            f"Meeko receptor preparation failed ({exc}); falling back to a "
            "minimal element-typed receptor. Pose accuracy will be lower — "
            "supply a complete protein PDB for the Meeko path.",
            stacklevel=2,
        )

    return _write_receptor_pdbqt_minimal(cleaned, out_path)


# AutoDock atom types for the elements that appear in cleaned protein receptors.
_AD_TYPE = {
    "C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "HD",
    "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "ZN": "Zn", "MG": "Mg", "CA": "Ca", "FE": "Fe", "MN": "Mn",
}


def _write_receptor_pdbqt_minimal(pdb_path: Path, out_path: Path) -> Path:
    """Write a rigid-receptor PDBQT directly from a cleaned PDB.

    PDBQT is column-strict, so the layout below is fixed by the format:
    name 13-16, altLoc 17, resName 18-20, chain 22, resSeq 23-26,
    x/y/z 31-54, occupancy 55-60, B-factor 61-66, charge 71-76, type 78-79.

    Partial charges are written as zero: Vina's own scoring function is
    steric/hydrophobic/H-bond based and does not read them (only the AutoDock4
    scoring function would).
    """
    from caddack.gnn.geometry import parse_pdb_atoms

    atoms = parse_pdb_atoms(pdb_path)
    if not atoms:
        raise ValueError(f"No atoms parsed from {pdb_path}")

    rows = []
    serial = 0
    for a in atoms:
        element = a.element.upper()
        if element == "H":
            continue  # rigid receptor is united-atom; polar H are implicit
        serial += 1
        ad = _AD_TYPE.get(element, "C")
        # PDB convention: single-letter elements are right-justified in col 14
        name = a.name if len(a.name) >= 4 else " " + a.name
        rows.append(
            f"ATOM  {serial:5d} {name:<4.4s}"          # 1-16
            f" {a.resname:>3.3s} {(a.chain or 'A'):1.1s}{a.resseq:4d}"  # 17-26
            f"    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"      # 27-54
            f"{1.00:6.2f}{0.00:6.2f}"                  # 55-66
            f"    {0.0:6.3f} {ad:<2s}"                 # 67-79
        )
    out_path.write_text("\n".join(rows) + "\nTER\nEND\n", encoding="utf-8")
    return out_path


# ------------------------------------------------------------------------ box
def box_from_coords(coords: Sequence[Sequence[float]], padding: float = 4.0,
                    min_size: float = 16.0) -> Box:
    """Axis-aligned box covering ``coords``, expanded by ``padding``."""
    if not len(coords):
        raise ValueError("No coordinates given")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
              (min(zs) + max(zs)) / 2)
    size = tuple(
        max(hi - lo + 2 * padding, min_size)
        for lo, hi in ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))
    )
    return Box(center=center, size=size)  # type: ignore[arg-type]


def box_from_reference_ligand(ligand_path: str | Path, padding: float = 4.0,
                              min_size: float = 16.0) -> Box:
    """Search box centred on a known ligand — the usual choice for re-docking."""
    from caddack.gnn.geometry import load_ligand

    atoms = load_ligand(ligand_path)
    if not atoms:
        raise ValueError(f"Could not read a ligand from {ligand_path}")
    return box_from_coords([(a.x, a.y, a.z) for a in atoms],
                           padding=padding, min_size=min_size)


def box_from_pocket_pdb(pocket_path: str | Path, padding: float = 2.0,
                        min_size: float = 16.0) -> Box:
    """Search box covering a pocket PDB (e.g. PDBbind's ``*_pocket.pdb``)."""
    from caddack.gnn.geometry import parse_pdb_atoms

    atoms = parse_pdb_atoms(pocket_path)
    if not atoms:
        raise ValueError(f"No atoms parsed from {pocket_path}")
    return box_from_coords([(a.x, a.y, a.z) for a in atoms],
                           padding=padding, min_size=min_size)
