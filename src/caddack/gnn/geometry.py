from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Atom element → atomic number (sufficient subset for organic/bio molecules)
_ELEMENT_Z: Dict[str, int] = {
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


def _element_from_name(name: str) -> str:
    """Infer element from PDB atom name when the element column is blank."""
    stripped = name.strip().lstrip("0123456789")
    return stripped[:2].upper() if len(stripped) >= 2 else stripped[:1].upper()


def parse_pdb_atoms(pdb_path: str | Path) -> List[PDBAtom]:
    """Fixed-width PDB ATOM/HETATM parser. No Biopython needed.

    Takes first MODEL only; skips alternate locations (altLoc != ' '/'A').
    Returns an empty list if the file is missing or unreadable.
    """
    atoms: List[PDBAtom] = []
    path = Path(pdb_path)
    if not path.exists():
        return atoms
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
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
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = int(line[22:26].strip() or "0")
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            raw_elem = line[76:78].strip().upper() if len(line) > 76 else ""
            element = raw_elem if raw_elem else _element_from_name(name)
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
            "Install with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem, AllChem


def load_ligand(ligand_path: str | Path) -> Optional[List[PDBAtom]]:
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
        elif suffix in (".mol2", ".mol"):
            mol = Chem.MolFromMolFile(str(path), removeHs=True)
        else:
            return None
    except Exception:
        return None

    if mol is None or not mol.GetNumConformers():
        return None

    conf = mol.GetConformer(0)
    atoms: List[PDBAtom] = []
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
    protein_atoms: List[PDBAtom],
    ligand_atoms: List[PDBAtom],
    cutoff: float = 6.0,
) -> List[PDBAtom]:
    """Return protein ATOM records within `cutoff` Å of any ligand atom."""
    pocket: List[PDBAtom] = []
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
    atomic_nums: List[int]        # [N] integer Z values
    positions: List[Tuple[float, float, float]]  # [N] (x,y,z)
    n_ligand: int                 # first n_ligand entries are ligand atoms
    n_pocket: int                 # remaining entries are pocket atoms


@dataclass
class ComplexExample:
    """Single protein–ligand complex for fusion model training."""
    pdb_id: str
    affinity: float               # e.g. pKd / pIC50
    ligand_smiles: Optional[str]  # may be None if ligand is sdf-only
    geo: GeometryRecord
    # ligand 2D graph arrays are built on-the-fly from ligand_smiles in the dataset loader


def _build_geo_record(
    ligand_atoms: List[PDBAtom],
    pocket_atoms: List[PDBAtom],
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
) -> Optional[ComplexExample]:
    """Load one protein–ligand complex from PDB + ligand file.

    Returns None if parsing yields fewer than 1 ligand or pocket atom.
    """
    protein_atoms = parse_pdb_atoms(pdb_path)
    ligand_atoms = load_ligand(ligand_path)
    if not ligand_atoms:
        return None
    pocket_atoms = extract_pocket(protein_atoms, ligand_atoms, cutoff=cutoff)
    if not pocket_atoms:
        return None

    geo = _build_geo_record(ligand_atoms, pocket_atoms)

    # Try to extract canonical SMILES from ligand file for Tower-1
    smiles = None
    try:
        Chem, _ = _require_rdkit()
        path = Path(ligand_path)
        if path.suffix.lower() == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
        else:
            mol = Chem.MolFromMolFile(str(path), removeHs=True)
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
) -> List[ComplexExample]:
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
    examples: List[ComplexExample] = []

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

    return examples
