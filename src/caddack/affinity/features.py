"""Features for protein-ligand affinity prediction.

One vector per complex, assembled from three sources:

* six physicochemical descriptors of the ligand,
* three size counts (ligand atoms, pocket atoms, their ratio),
* Vina's six interaction terms,
* one hundred contact counts, split by element-class pair and distance shell.

The last two describe atom *pairs* across the interface, and that is what makes
the set work. A two-tower graph network given only atoms reached r 0.644 on a
PDBbind scaffold split; a random forest on these 115 numbers reaches 0.696 on the
same split, and the network did not catch up when handed the same features. The
pairwise structure appears to be the thing worth supplying rather than hoping a
network discovers it from a few thousand examples.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from caddack.docking.scoring import (
    CONTACT_FEAT_DIM,
    TERM_NAMES,
    contact_feature_names,
    contact_features,
    vina_terms,
)

DESCRIPTOR_NAMES = ("MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds")
SIZE_NAMES = ("n_ligand_atoms", "n_pocket_atoms", "pocket_per_ligand_atom")
# 8 A, matching Vina's own interaction cutoff and the outermost contact shell.
DEFAULT_CUTOFF = 8.0


def feature_names() -> list[str]:
    return [*DESCRIPTOR_NAMES, *SIZE_NAMES, *TERM_NAMES, "vina_inter",
            *contact_feature_names()]


FEATURE_DIM = len(DESCRIPTOR_NAMES) + len(SIZE_NAMES) + len(TERM_NAMES) + 1 + CONTACT_FEAT_DIM


def features_from_atoms(
    atomic_nums: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    n_ligand: int,
    smiles: str,
) -> np.ndarray | None:
    """Build the feature vector from coordinates plus the ligand's SMILES.

    Ligand atoms come first in ``atomic_nums`` and ``positions``. Returns None if
    the SMILES cannot be parsed, since the descriptors come from it.
    """
    from caddack.qsar.descriptors import mol_to_basic_features, smiles_to_mol_clean

    mol = smiles_to_mol_clean(smiles) if smiles else None
    if mol is None:
        return None

    desc = mol_to_basic_features(mol)
    n_lig_heavy = mol.GetNumAtoms()
    n_pocket = len(atomic_nums) - n_ligand

    terms = vina_terms(atomic_nums, positions, n_ligand=n_ligand)
    contacts = contact_features(atomic_nums, positions, n_ligand=n_ligand)

    return np.concatenate([
        [desc[k] for k in DESCRIPTOR_NAMES],
        [n_lig_heavy, n_pocket, n_pocket / max(n_lig_heavy, 1)],
        [terms[k] for k in TERM_NAMES],
        [terms["vina_inter"]],
        contacts,
    ]).astype(np.float64)


def features_from_files(
    pocket_pdb: str | Path,
    ligand_file: str | Path,
    cutoff: float = DEFAULT_CUTOFF,
) -> np.ndarray | None:
    """Build the feature vector straight from a pocket PDB and a ligand file.

    The pocket is trimmed to atoms within ``cutoff`` of any ligand atom. Unlike
    the cached geometry used for the graph models, HETATM records are kept, so
    cofactors and metals are included -- dropping them measurably weakens the
    interaction terms.
    """
    from rdkit import Chem

    from caddack.gnn.geometry import load_ligand, parse_pdb_atoms

    protein = parse_pdb_atoms(pocket_pdb)
    ligand = load_ligand(ligand_file)
    if not protein or not ligand:
        return None

    lig_pos = np.array([[a.x, a.y, a.z] for a in ligand])
    lig_z = [a.atomic_num for a in ligand]
    pro_pos = np.array([[a.x, a.y, a.z] for a in protein])
    pro_z = np.array([a.atomic_num for a in protein])

    near = np.sqrt(((pro_pos[:, None, :] - lig_pos[None, :, :]) ** 2).sum(-1)).min(1) <= cutoff

    suffix = Path(ligand_file).suffix.lower()
    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_file), removeHs=True)
    elif suffix == ".sdf":
        mol = next((m for m in Chem.SDMolSupplier(str(ligand_file), removeHs=True)
                    if m is not None), None)
    else:
        mol = Chem.MolFromMolFile(str(ligand_file), removeHs=True)
    if mol is None:
        return None

    return features_from_atoms(
        list(lig_z) + list(pro_z[near]),
        np.vstack([lig_pos, pro_pos[near]]),
        n_ligand=len(lig_z),
        smiles=Chem.MolToSmiles(mol),
    )
