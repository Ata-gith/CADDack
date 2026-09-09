"""AutoDock Vina docking, wrapped around the official Python bindings.

CADDack does not implement a pose search of its own — Vina's is mature and well
validated. What this module adds is the glue: preparing inputs from a SMILES or
a PDB, defining the box, running the search, and returning poses in a form the
rest of the package (in particular the affinity model) can consume.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from caddack.docking.prepare import (
    Box,
    box_from_reference_ligand,
    ligand_pdbqt_from_smiles,
    receptor_pdbqt_from_pdb,
)


@dataclass
class DockedPose:
    """One docked pose.

    ``score`` is Vina's predicted affinity in kcal/mol — more negative is
    stronger. It is an empirical scoring function, not a measured constant.
    """

    rank: int
    score: float
    rmsd_lb: float
    rmsd_ub: float
    pdbqt: str

    @property
    def coords(self) -> list[tuple[float, float, float]]:
        """Heavy-atom coordinates parsed out of the pose PDBQT."""
        out: list[tuple[float, float, float]] = []
        for line in self.pdbqt.splitlines():
            if line[:6].strip() in ("ATOM", "HETATM"):
                try:
                    out.append((float(line[30:38]), float(line[38:46]),
                                float(line[46:54])))
                except ValueError:
                    continue
        return out


def _require_vina():
    try:
        from vina import Vina
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "AutoDock Vina is required for docking. Install with `pip install vina`."
        ) from exc
    return Vina


def dock_pdbqt(receptor_pdbqt: str | Path, ligand_pdbqt: str | Path, box: Box,
               exhaustiveness: int = 8, n_poses: int = 9,
               seed: int = 42, verbosity: int = 0) -> list[DockedPose]:
    """Dock a prepared ligand into a prepared receptor.

    ``exhaustiveness`` trades runtime for search thoroughness (Vina's default is
    8; raise it for flexible ligands).
    """
    Vina = _require_vina()

    v = Vina(sf_name="vina", seed=seed, verbosity=verbosity)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(center=list(box.center), box_size=list(box.size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    # energies(): rows of [total, inter, intra, torsion, intra_best] and, once
    # poses exist, Vina reports RMSD to the best pose in the last two columns.
    energies = v.energies(n_poses=n_poses)
    blocks = _split_models(v.poses(n_poses=n_poses))

    poses: list[DockedPose] = []
    for i, block in enumerate(blocks):
        row = energies[i] if i < len(energies) else []
        score = float(row[0]) if len(row) else float("nan")
        rmsd_lb = float(row[-2]) if len(row) >= 2 else 0.0
        rmsd_ub = float(row[-1]) if len(row) >= 1 else 0.0
        poses.append(DockedPose(rank=i + 1, score=score, rmsd_lb=rmsd_lb,
                                rmsd_ub=rmsd_ub, pdbqt=block))
    return poses


def _split_models(pdbqt_text: str) -> list[str]:
    """Split a multi-MODEL PDBQT string into one block per pose."""
    blocks, current = [], []
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                blocks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def dock_smiles(receptor_pdb: str | Path, smiles: str,
                box: Box | None = None,
                reference_ligand: str | Path | None = None,
                workdir: str | Path | None = None,
                exhaustiveness: int = 8, n_poses: int = 9, seed: int = 42,
                keep_hetatm: Sequence[str] | None = None) -> list[DockedPose]:
    """Dock a SMILES string into a receptor PDB, end to end.

    Give either an explicit ``box`` or a ``reference_ligand`` to centre on.
    Intermediate PDBQT files are written to ``workdir`` (a temp dir by default)
    so a failed run can be inspected.

    Note that Vina samples position, orientation and acyclic torsions but keeps
    **ring conformations fixed** at whatever the input conformer has. For ligands
    with flexible or macrocyclic rings, generate several ring conformers and dock
    each rather than relying on the single conformer embedded here.

    ``exhaustiveness`` is Vina's default of 8; 32 is noticeably more reliable at
    roughly four times the runtime.
    """
    if box is None:
        if reference_ligand is None:
            raise ValueError(
                "Provide either box= or reference_ligand= to define the search box"
            )
        box = box_from_reference_ligand(reference_ligand)

    if workdir is None:
        import tempfile
        workdir = tempfile.mkdtemp(prefix="caddack_dock_")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = receptor_pdbqt_from_pdb(
        receptor_pdb, workdir / "receptor.pdbqt", keep_hetatm=keep_hetatm)
    ligand_pdbqt = ligand_pdbqt_from_smiles(
        smiles, workdir / "ligand.pdbqt", seed=seed)

    return dock_pdbqt(receptor_pdbqt, ligand_pdbqt, box,
                      exhaustiveness=exhaustiveness, n_poses=n_poses, seed=seed)


def pose_to_mol(pose: DockedPose):
    """Convert a docked pose to an RDKit Mol (bond orders restored by Meeko).

    A pose PDBQT is not valid PDB — it carries AutoDock types and a charge
    column — so RDKit's PDB reader cannot be used directly.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Meeko is required to convert poses to RDKit molecules. "
            "Install with `pip install meeko`."
        ) from exc

    block = pose.pdbqt
    if not block.lstrip().startswith("MODEL"):
        block = f"MODEL 1\n{block}\nENDMDL\n"
    pdbqt_mol = PDBQTMolecule(block, skip_typing=True)
    mols = [m for m in RDKitMolCreate.from_pdbqt_mol(pdbqt_mol) if m is not None]
    if not mols:
        raise ValueError("Could not rebuild an RDKit molecule from this pose")
    return mols[0]


def pose_rmsd(pose: DockedPose, reference_mol) -> float:
    """Symmetry-aware RMSD between a docked pose and a reference molecule.

    This is the standard re-docking success metric: below 2 A is conventionally
    counted as reproducing the crystal pose. Uses RDKit's ``CalcRMS``, which
    matches atoms by substructure and accounts for topological symmetry, so it
    does not depend on atom ordering. Hydrogens are ignored.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    probe = Chem.RemoveHs(pose_to_mol(pose))
    ref = Chem.RemoveHs(reference_mol)
    return float(rdMolAlign.CalcRMS(probe, ref))
