"""Features for protein-ligand affinity prediction.

One vector per complex, assembled from:

* six physicochemical descriptors of the ligand,
* the ligand's heavy-atom count,
* Vina's six interaction terms,
* one hundred contact counts, split by element-class pair and distance shell,
* optionally, ECFP4 fingerprint bits describing the ligand alone.

The contact counts and Vina terms describe atom *pairs* across the interface,
and that is what makes the set work. A two-tower graph network given only atoms
reached r 0.644 on a PDBbind scaffold split; a random forest on the 113
structural numbers reaches 0.696 on the same split, and the network did not
catch up when handed the same features. The pairwise structure appears to be the
thing worth supplying rather than hoping a network discovers it from a few
thousand examples.

The fingerprint is the complement to that. Contact counts say what the interface
looks like but discard which molecule made it, so two unrelated scaffolds
producing the same contact pattern are indistinguishable. Adding 1024 bits lifts
r from 0.686 to 0.704 over five scaffold splits (paired t = 5.3, p = 0.006),
while the bits alone reach only 0.524 -- they are complementary rather than
redundant. Note that this gain needs every feature available at each split:
``max_features="sqrt"`` samples 34 of 1137 columns, which rarely includes the
informative contacts, and the gain disappears entirely (-0.001 +- 0.004).
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
# Only the ligand's size. Pocket atom count and the pocket/ligand ratio were in
# this set and came out as the second most important feature of 115, but they
# were both redundant -- removing them moved r by +0.003 and MAE by -0.005 over
# three splits -- and contaminated: PDBbind structures carry hydrogens
# inconsistently, which inflated the pocket count by about 24% at random. A
# feature the model leans on should not depend on what the depositor chose to
# model.
SIZE_NAMES = ("n_ligand_atoms",)
# 8 A, matching Vina's own interaction cutoff and the outermost contact shell.
DEFAULT_CUTOFF = 8.0


# ECFP radius 2, which is ECFP*4* -- the number in the name is the diameter.
FP_RADIUS = 2
DEFAULT_FP_BITS = 1024


def feature_names(fp_bits: int = 0) -> list[str]:
    names = [*DESCRIPTOR_NAMES, *SIZE_NAMES, *TERM_NAMES, "vina_inter",
             *contact_feature_names()]
    names += [f"ECFP4_{i}" for i in range(fp_bits)]
    return names


STRUCTURAL_DIM = (len(DESCRIPTOR_NAMES) + len(SIZE_NAMES) + len(TERM_NAMES)
                  + 1 + CONTACT_FEAT_DIM)
FEATURE_DIM = STRUCTURAL_DIM


def feature_dim(fp_bits: int = 0) -> int:
    return STRUCTURAL_DIM + fp_bits


def features_from_atoms(
    atomic_nums: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    n_ligand: int,
    smiles: str,
    fp_bits: int = 0,
) -> np.ndarray | None:
    """Build the feature vector from coordinates plus the ligand's SMILES.

    Ligand atoms come first in ``atomic_nums`` and ``positions``. Returns None if
    the SMILES cannot be parsed, since the descriptors come from it.

    ``fp_bits`` appends that many ECFP4 bits describing the ligand alone. The
    contact counts say what the interface looks like but discard which molecule
    made it, so two unrelated scaffolds producing the same contact pattern are
    indistinguishable without them. Adding 1024 bits lifts r from 0.686 to 0.704
    over five scaffold splits (paired t = 5.3, p = 0.006). Zero keeps the
    structural features alone, which is what earlier models shipped.
    """
    from caddack.qsar.descriptors import (
        mol_to_basic_features,
        mol_to_ecfp_array,
        smiles_to_mol_clean,
    )

    mol = smiles_to_mol_clean(smiles) if smiles else None
    if mol is None:
        return None

    # The interaction features reach 8 A. A pocket trimmed closer than that
    # silently empties the outer shells, and the model was trained on full ones:
    # passing the 6 A cached geometry roughly halves the 6-8 A contact counts.
    # Warn rather than fail, since a genuinely small pocket is also possible.
    pos_arr = np.asarray(positions, dtype=float)
    if n_ligand < len(pos_arr):
        reach = np.linalg.norm(
            pos_arr[n_ligand:][:, None, :] - pos_arr[:n_ligand][None, :, :],
            axis=-1).min(axis=1).max()
        if reach < DEFAULT_CUTOFF - 0.5:
            import warnings

            warnings.warn(
                f"pocket reaches only {reach:.1f} A from the ligand but the "
                f"interaction features extend to {DEFAULT_CUTOFF:.0f} A; the outer "
                "contact shells will be empty and the prediction will not match "
                "how the model was trained",
                RuntimeWarning, stacklevel=2,
            )

    desc = mol_to_basic_features(mol)
    n_lig_heavy = mol.GetNumAtoms()

    terms = vina_terms(atomic_nums, positions, n_ligand=n_ligand)
    contacts = contact_features(atomic_nums, positions, n_ligand=n_ligand)

    blocks = [
        [desc[k] for k in DESCRIPTOR_NAMES],
        [n_lig_heavy],
        [terms[k] for k in TERM_NAMES],
        [terms["vina_inter"]],
        contacts,
    ]
    if fp_bits:
        # From the same cleaned molecule the descriptors came from, so the salt
        # stripping that mol_to_basic_features saw is the stripping the
        # fingerprint sees. The browser strips salts separately before
        # fingerprinting for exactly this reason.
        blocks.append(np.unpackbits(
            mol_to_ecfp_array(mol, radius=FP_RADIUS, n_bits=fp_bits)))
    return np.concatenate(blocks).astype(np.float64)


def features_from_files(
    pocket_pdb: str | Path,
    ligand_file: str | Path,
    cutoff: float = DEFAULT_CUTOFF,
    fp_bits: int = 0,
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
        fp_bits=fp_bits,
    )
