"""Molecular docking: pose generation with AutoDock Vina.

The affinity model in ``caddack.gnn`` scores a protein-ligand pose. Docking is
what supplies that pose when no crystal structure exists, which is the usual
case for a new molecule.

Typical use::

    from caddack.docking import dock_smiles, box_from_reference_ligand

    box = box_from_reference_ligand("data/1q1m_ligand.mol2")
    poses = dock_smiles("data/1q1m_protein.pdb", "CC(=O)Oc1ccccc1C(=O)O", box=box)
    print(poses[0].score)   # kcal/mol, more negative is stronger

Imports stay light: RDKit, Meeko and Vina are only required when a function that
needs them is called.
"""
from caddack.docking.prepare import (
    Box,
    box_from_coords,
    box_from_pocket_pdb,
    box_from_reference_ligand,
    clean_receptor_pdb,
    embed_3d,
    ligand_pdbqt_from_mol,
    ligand_pdbqt_from_smiles,
    receptor_pdbqt_from_pdb,
)
from caddack.docking.vina import (
    DockedPose,
    dock_pdbqt,
    dock_smiles,
    pose_rmsd,
    pose_to_mol,
)

__all__ = [
    "Box",
    "DockedPose",
    "box_from_coords",
    "box_from_pocket_pdb",
    "box_from_reference_ligand",
    "clean_receptor_pdb",
    "dock_pdbqt",
    "dock_smiles",
    "embed_3d",
    "ligand_pdbqt_from_mol",
    "ligand_pdbqt_from_smiles",
    "pose_rmsd",
    "pose_to_mol",
    "receptor_pdbqt_from_pdb",
]
