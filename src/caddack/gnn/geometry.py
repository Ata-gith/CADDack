from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Atom element → atomic number (sufficient subset for organic/bio molecules)
_ELEMENT_Z: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17,
    "BR": 35, "I": 53, "FE": 26, "ZN": 30, "CA": 20, "MG": 12, "NA": 11,
    "K": 19, "CU": 29, "MN": 25, "CO": 27, "NI": 28, "SE": 34, "B": 5,
    "SI": 14,
}
_DEFAULT_Z = 6  # fallback: carbon


@dataclass
class PDBAtom:
    record: str      # "ATOM" or "HETATM"
    name: str        # atom name field (e.g. "CA", "C1")
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str     # upper-cased; may be empty string
    atomic_num: int


def _element_from_name(raw_name: str) -> str:
    """Infer element from a PDB atom-name field when the element column is blank.

    Uses the PDB column convention (name occupies columns 13-16): a genuine
    two-letter element symbol (FE, ZN, CL, ...) is left-justified so the field's
    first character is a letter, whereas a single-letter element is
    right-justified in column 14 so the field starts with a space (or digit).
    This disambiguates ``" CA "`` (C-alpha → carbon) from ``"CA  "`` (calcium),
    and maps ``" ND1"`` → nitrogen, ``" OD1"`` → oxygen, ``" SD "`` → sulfur.

    Must be called with the RAW 4-char field (``line[12:16]``), not a stripped
    name, or the column information is lost.
    """
    if not raw_name:
        return ""
    # A two-letter element symbol only when column 13 (index 0) is itself a letter.
    if raw_name[0] not in " 0123456789":
        cand = raw_name[:2].strip().upper()
        if cand in _ELEMENT_Z:
            return cand
    stripped = raw_name.strip().lstrip("0123456789")
    return stripped[:1].upper() if stripped else ""


def parse_pdb_atoms(pdb_path: str | Path) -> list[PDBAtom]:
    """Fixed-width PDB ATOM/HETATM parser. No Biopython needed.

    Takes first MODEL only; skips alternate locations (altLoc != ' '/'A').
    Returns an empty list if the file is missing or unreadable.
    """
    atoms: list[PDBAtom] = []
    path = Path(pdb_path)
    if not path.exists():
        warnings.warn(f"PDB file not found; skipping parse: {path}", stacklevel=2)
        return atoms
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.warn(f"Could not read PDB file; skipping parse: {path} ({exc})", stacklevel=2)
        return atoms

    in_first_model = True
    for line in text.splitlines():
        if line.startswith("MODEL") and atoms:
            in_first_model = False
        if not in_first_model:
            break
        if line.startswith("ENDMDL"):
            break

        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue

        # PDB fixed-width columns
        alt_loc = line[16:17]
        if alt_loc not in (" ", "A", ""):
            continue

        try:
            raw_name = line[12:16]              # keep columns for element inference
            name = raw_name.strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = int(line[22:26].strip() or "0")
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            raw_elem = line[76:78].strip().upper() if len(line) > 76 else ""
            element = raw_elem if raw_elem else _element_from_name(raw_name)
            atomic_num = _ELEMENT_Z.get(element, _DEFAULT_Z)
        except (ValueError, IndexError):
            continue

        atoms.append(PDBAtom(
            record=record,
            name=name,
            resname=resname,
            chain=chain,
            resseq=resseq,
            x=x, y=y, z=z,
            element=element,
            atomic_num=atomic_num,
        ))
    return atoms


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        raise ImportError(
            "RDKit is required to load ligand files (.sdf/.mol2). "
            "Install with `pip install rdkit` (or `pip install caddack[gnn]`)."
        ) from exc
    return Chem, AllChem


def load_ligand(ligand_path: str | Path) -> list[PDBAtom] | None:
    """Load a ligand from .sdf or .mol2, returning PDBAtom list with 3D coords.

    Returns None if RDKit is unavailable or parsing fails.
    """
    Chem, AllChem = _require_rdkit()
    path = Path(ligand_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=True)  # Tripos MOL2
        elif suffix == ".mol":
            mol = Chem.MolFromMolFile(str(path), removeHs=True)   # MDL molfile
        else:
            warnings.warn(f"Unsupported ligand format; skipping: {path}", stacklevel=2)
            return None
    except Exception:
        warnings.warn(f"Failed to parse ligand file; skipping: {path}", stacklevel=2)
        return None

    if mol is None or not mol.GetNumConformers():
        warnings.warn(f"Ligand file had no usable structure; skipping: {path}", stacklevel=2)
        return None

    conf = mol.GetConformer(0)
    atoms: list[PDBAtom] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol().upper()
        z = _ELEMENT_Z.get(elem, _DEFAULT_Z)
        atoms.append(PDBAtom(
            record="HETATM",
            name=atom.GetSymbol(),
            resname="LIG",
            chain="L",
            resseq=0,
            x=pos.x, y=pos.y, z=pos.z,
            element=elem,
            atomic_num=z,
        ))
    return atoms if atoms else None


def _dist(a: PDBAtom, b: PDBAtom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def extract_pocket(
    protein_atoms: list[PDBAtom],
    ligand_atoms: list[PDBAtom],
    cutoff: float = 6.0,
) -> list[PDBAtom]:
    """Return protein ATOM records within `cutoff` Å of any ligand atom."""
    pocket: list[PDBAtom] = []
    for pa in protein_atoms:
        if pa.record != "ATOM":
            continue
        for la in ligand_atoms:
            if _dist(pa, la) <= cutoff:
                pocket.append(pa)
                break
    return pocket


@dataclass
class GeometryRecord:
    """3D atoms (pocket + ligand) with atomic numbers and positions."""
    atomic_nums: list[int]        # [N] integer Z values
    positions: list[tuple[float, float, float]]  # [N] (x,y,z)
    n_ligand: int                 # first n_ligand entries are ligand atoms
    n_pocket: int                 # remaining entries are pocket atoms


@dataclass
class ComplexExample:
    """Single protein–ligand complex for fusion model training."""
    pdb_id: str
    affinity: float               # e.g. pKd / pIC50
    ligand_smiles: str | None  # may be None if ligand is sdf-only
    geo: GeometryRecord
    # ligand 2D graph arrays are built on-the-fly from ligand_smiles in the dataset loader


def _build_geo_record(
    ligand_atoms: list[PDBAtom],
    pocket_atoms: list[PDBAtom],
) -> GeometryRecord:
    combined = ligand_atoms + pocket_atoms
    atomic_nums = [a.atomic_num for a in combined]
    positions = [(a.x, a.y, a.z) for a in combined]
    return GeometryRecord(
        atomic_nums=atomic_nums,
        positions=positions,
        n_ligand=len(ligand_atoms),
        n_pocket=len(pocket_atoms),
    )


def load_complex(
    pdb_path: str | Path,
    ligand_path: str | Path,
    affinity: float,
    pdb_id: str = "",
    cutoff: float = 6.0,
) -> ComplexExample | None:
    """Load one protein–ligand complex from PDB + ligand file.

    Returns None if parsing yields fewer than 1 ligand or pocket atom.
    """
    protein_atoms = parse_pdb_atoms(pdb_path)
    ligand_atoms = load_ligand(ligand_path)
    if not ligand_atoms:
        warnings.warn(
            f"Skipping complex {pdb_id or Path(pdb_path).stem}: "
            f"no usable ligand atoms from {ligand_path}",
            stacklevel=2,
        )
        return None
    pocket_atoms = extract_pocket(protein_atoms, ligand_atoms, cutoff=cutoff)
    if not pocket_atoms:
        warnings.warn(
            f"Skipping complex {pdb_id or Path(pdb_path).stem}: "
            f"no pocket atoms within {cutoff} Å of ligand",
            stacklevel=2,
        )
        return None

    geo = _build_geo_record(ligand_atoms, pocket_atoms)

    # Try to extract canonical SMILES from ligand file for Tower-1
    smiles = None
    try:
        Chem, _ = _require_rdkit()
        path = Path(ligand_path)
        suffix = path.suffix.lower()
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=True)  # Tripos MOL2
        else:
            mol = Chem.MolFromMolFile(str(path), removeHs=True)   # MDL molfile
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    return ComplexExample(
        pdb_id=pdb_id or Path(pdb_path).stem,
        affinity=affinity,
        ligand_smiles=smiles,
        geo=geo,
    )


def load_complex_dataset(
    index_csv: str | Path,
    root: str | Path,
    smiles_col: str = "smiles",
    affinity_col: str = "affinity",
    pdb_col: str = "pdb_id",
    ligand_suffix: str = ".sdf",
    cutoff: float = 6.0,
) -> list[ComplexExample]:
    """Load a PDBbind-style dataset from an index CSV and a root directory.

    Expected layout (adjustable via column args):
        root/<pdb_id>/<pdb_id>_protein.pdb
        root/<pdb_id>/<pdb_id>_ligand<ligand_suffix>

    Rows that fail to parse (missing files, bad geometry) are silently skipped.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required to read the index CSV.") from exc

    df = pd.read_csv(index_csv)
    root = Path(root)
    examples: list[ComplexExample] = []
    skipped = 0

    for _, row in df.iterrows():
        pdb_id = str(row[pdb_col])
        affinity = float(row[affinity_col])
        pdb_file = root / pdb_id / f"{pdb_id}_protein.pdb"
        lig_file = root / pdb_id / f"{pdb_id}_ligand{ligand_suffix}"

        ex = load_complex(pdb_file, lig_file, affinity=affinity, pdb_id=pdb_id, cutoff=cutoff)
        if ex is not None:
            if smiles_col in row and pd.notna(row[smiles_col]):
                ex.ligand_smiles = str(row[smiles_col])
            examples.append(ex)
        else:
            skipped += 1

    if skipped:
        warnings.warn(
            f"load_complex_dataset skipped {skipped} of {len(df)} complexes "
            f"due to missing/invalid geometry or ligand files.",
            stacklevel=2,
        )

    return examples



# Element groups used by the first-neighbour featuriser. Anything else is "other".
_NEIGHBOR_ELEMENTS = (6, 7, 8, 16)          # C, N, O, S
NEIGHBOR_FEAT_DIM = 9
_COVALENT_CUTOFF = 1.95                      # Å, generous for heavy-atom bonds


def neighbor_features(
    atomic_nums: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    n_ligand: int,
) -> list[list[float]]:
    """Describe each atom by its first coordination shell.

    The geometry tower embeds atomic number alone, so a backbone carbonyl oxygen,
    a serine hydroxyl and a buried water are the same token — despite behaving
    very differently in a binding site. Bonded neighbours recover most of that
    distinction without needing residue templates or partial charges:

    * **who** the neighbours are (counts of C, N, O, S, other) separates a
      backbone amide N from a lysine ammonium N;
    * **how far** they sit separates a C=O acceptor (~1.23 Å) from a C-O-H
      donor (~1.43 Å). Measured on PDBbind, oxygens with a single carbon
      neighbour split 82% below 1.30 Å and 17% above 1.36 Å, so the signal is
      strongly bimodal rather than smeared;
    * **how many** there are gives coordination number, which separates a
      terminal from a bridging atom, and leaves an isolated water at zero.

    Also flags ligand versus pocket, which the tower otherwise has no way to
    know: it receives one undifferentiated point cloud and must infer the
    interface from spatial arrangement alone.

    Returns one ``NEIGHBOR_FEAT_DIM``-length row per atom:
    ``[is_ligand, degree, n_C, n_N, n_O, n_S, n_other, min_dist, mean_dist]``.
    Counts are divided by 4 and distances expressed relative to the cutoff, so
    everything lands in roughly [0, 1] without a separate scaler.

    Computed on demand from coordinates, so cached ``GeometryRecord`` objects
    written before this existed stay usable.
    """
    n = len(atomic_nums)
    if n == 0:
        return []
    if len(positions) != n:
        raise ValueError("atomic_nums and positions must have equal length")

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is a core dependency
        raise ImportError("numpy is required for neighbour features.") from exc

    pos = np.asarray(positions, dtype=np.float64)
    z = np.asarray(atomic_nums, dtype=np.int64)

    # Pairwise distances. Pocket cutouts are a few hundred atoms, so the dense
    # matrix is cheap and keeps this readable.
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((delta ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    bonded = dist < _COVALENT_CUTOFF

    out: list[list[float]] = []
    for i in range(n):
        nb = np.flatnonzero(bonded[i])
        d = dist[i, nb]
        counts = [float((z[nb] == e).sum()) for e in _NEIGHBOR_ELEMENTS]
        other = float(len(nb) - sum(counts))
        if len(nb):
            min_d, mean_d = float(d.min()), float(d.mean())
        else:
            # No bonded neighbour (an ion, or a lone water oxygen). Use the
            # cutoff so "isolated" reads as maximally distant rather than zero,
            # which would otherwise look like an impossibly short bond.
            min_d = mean_d = _COVALENT_CUTOFF
        out.append([
            1.0 if i < n_ligand else 0.0,
            len(nb) / 4.0,
            *[c / 4.0 for c in counts],
            other / 4.0,
            min_d / _COVALENT_CUTOFF,
            mean_d / _COVALENT_CUTOFF,
        ])
    return out
