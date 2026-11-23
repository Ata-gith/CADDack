"""AutoDock Vina's scoring terms, computed directly from coordinates.

Vina's empirical score is a weighted sum of five functions of the *surface*
distance between atom pairs -- the centre distance minus both van der Waals
radii (Trott & Olson, J. Comput. Chem. 31:455, 2010). Evaluating them here,
rather than shelling out to Vina, avoids a PDBQT round trip and returns the
terms separately instead of only their weighted total, which is what makes them
usable as model features.

Why this exists: the affinity model was found to be reproducible by nine scalar
descriptors, i.e. it was not using the structure at all. These terms are an
explicit interaction energy -- they cannot be computed from an atom inventory,
only from the geometry -- so they test whether the model benefits from being
handed the physics rather than being asked to discover it.

Two caveats carried from Vina itself. There is **no electrostatic term**: formal
charge and pH are invisible to this function, as they are to Vina. And hydrogens
are ignored, the typing being united-atom, which is convenient here because
PDBbind pocket files carry only partial protonation.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# X-Score van der Waals radii, as used by Vina. Anything unlisted is treated as
# a generic metal, which Vina also does.
_VDW_RADII = {
    6: 1.9, 7: 1.8, 8: 1.7, 15: 2.1, 16: 2.0,
    9: 1.5, 17: 1.8, 35: 2.0, 53: 2.2,
}
_METAL_RADIUS = 1.2
_HALOGENS = frozenset({9, 17, 35, 53})
_POLAR = frozenset({7, 8})           # N, O: donors and acceptors, united-atom
_PAIR_CUTOFF = 8.0                   # A, Vina's interaction cutoff
_BOND_CUTOFF = 1.95                  # A, heavy-atom covalent bond

# Weights from the published vina scoring function.
WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "hbond": -0.587439,
}
_ROT_WEIGHT = 0.058459

TERM_NAMES = ("gauss1", "gauss2", "repulsion", "hydrophobic", "hbond")


def _radii(z: np.ndarray) -> np.ndarray:
    return np.array([_VDW_RADII.get(int(v), _METAL_RADIUS) for v in z], dtype=float)


def _hydrophobic_mask(z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Carbon not bonded to N or O, plus halogens -- Vina's hydrophobic typing.

    A carbon next to a heteroatom is polarised and does not count, which is why
    bonded neighbours have to be worked out rather than reading the element.
    """
    out = np.isin(z, list(_HALOGENS))
    carbons = np.flatnonzero(z == 6)
    if carbons.size:
        d = np.linalg.norm(pos[carbons, None, :] - pos[None, :, :], axis=-1)
        d[np.arange(carbons.size), carbons] = np.inf          # ignore self
        polar_neighbour = ((d < _BOND_CUTOFF) & np.isin(z, list(_POLAR))[None, :]).any(1)
        out[carbons] = ~polar_neighbour
    return out


def vina_terms(
    atomic_nums: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    n_ligand: int,
    n_rotatable: int | None = None,
) -> dict[str, float]:
    """Vina's five interaction terms, summed over protein-ligand atom pairs.

    ``atomic_nums`` and ``positions`` describe one complex with the ligand's
    atoms first, matching :class:`caddack.gnn.geometry.GeometryRecord`. Only
    pairs that cross the interface contribute; ligand-ligand and pocket-pocket
    pairs are intramolecular and are not part of the intermolecular score.

    Returns the five raw terms, their weighted sum (``vina_inter``), and -- when
    ``n_rotatable`` is given -- the torsion-corrected estimate Vina reports as a
    binding free energy in kcal/mol.
    """
    z = np.asarray(atomic_nums, dtype=int)
    pos = np.asarray(positions, dtype=float)
    if z.shape[0] != pos.shape[0]:
        raise ValueError("atomic_nums and positions must have equal length")

    # Vina is united-atom and ignores hydrogens. Keeping them would also give
    # them the fallback metal radius, since H has no entry in the radius table,
    # which distorts every surface distance they take part in. PDBbind pocket
    # files carry polar hydrogens only, so this has to be handled explicitly.
    heavy = z != 1
    if not heavy.all():
        n_ligand = int(heavy[:n_ligand].sum())
        z, pos = z[heavy], pos[heavy]
    zero = dict.fromkeys(TERM_NAMES, 0.0) | {"vina_inter": 0.0, "n_contacts": 0.0}
    if n_ligand <= 0 or n_ligand >= len(z):
        return zero | ({"vina_score": 0.0} if n_rotatable is not None else {})

    lig, poc = slice(0, n_ligand), slice(n_ligand, None)
    r = np.linalg.norm(pos[lig][:, None, :] - pos[poc][None, :, :], axis=-1)
    within = r <= _PAIR_CUTOFF
    if not within.any():
        return zero | ({"vina_score": 0.0} if n_rotatable is not None else {})

    radii = _radii(z)
    # Surface distance: centre separation less both van der Waals radii.
    d = r - (radii[lig][:, None] + radii[poc][None, :])
    d = d[within]

    hydro = _hydrophobic_mask(z, pos)
    hydro_pair = (hydro[lig][:, None] & hydro[poc][None, :])[within]
    # United-atom H-bonding: any N/O facing any N/O. Without hydrogens the
    # donor/acceptor roles cannot be separated, which is Vina's own approximation.
    polar = np.isin(z, list(_POLAR))
    polar_pair = (polar[lig][:, None] & polar[poc][None, :])[within]

    terms = {
        "gauss1": float(np.exp(-((d / 0.5) ** 2)).sum()),
        "gauss2": float(np.exp(-(((d - 3.0) / 2.0) ** 2)).sum()),
        "repulsion": float((np.where(d < 0, d, 0.0) ** 2).sum()),
        "hydrophobic": float(np.clip(1.5 - d, 0.0, 1.0)[hydro_pair].sum()),
        "hbond": float(np.clip(-d / 0.7, 0.0, 1.0)[polar_pair].sum()),
    }
    terms["vina_inter"] = float(sum(WEIGHTS[k] * terms[k] for k in TERM_NAMES))
    terms["n_contacts"] = float(within.sum())

    if n_rotatable is not None:
        # Vina divides the intermolecular score by the torsion penalty rather
        # than adding it, so the correction scales with the interaction.
        terms["vina_score"] = terms["vina_inter"] / (1.0 + _ROT_WEIGHT * n_rotatable)
    return terms


# --- distance-binned contact counts -------------------------------------------
# Vina's terms collapse every atom pair into five numbers. Published scoring
# models that rank well on this task instead count contacts separately for each
# combination of atom types and distance shell, letting the model weigh a
# carbon-carbon contact at 3.5 A differently from a nitrogen-oxygen one at 6 A.
# That is structure a network would otherwise have to discover from a few
# thousand examples.

# Element classes, coarse enough that 2,959 training complexes can populate the
# bins. Everything unlisted (metals, boron, selenium) lands in "other".
_ATOM_CLASSES: tuple[tuple[str, frozenset[int]], ...] = (
    ("C", frozenset({6})),
    ("N", frozenset({7})),
    ("O", frozenset({8})),
    ("X", frozenset({16, 15, 9, 17, 35, 53})),   # S, P, halogens
)
_N_CLASSES = len(_ATOM_CLASSES) + 1              # + "other"
# Shell edges in angstroms. The first captures hydrogen-bond and close-contact
# range, the last reaches past Vina's own 8 A cutoff.
CONTACT_SHELLS: tuple[float, ...] = (3.5, 4.5, 6.0, 8.0)
CONTACT_FEAT_DIM = _N_CLASSES * _N_CLASSES * len(CONTACT_SHELLS)


def _class_index(z: np.ndarray) -> np.ndarray:
    idx = np.full(z.shape, _N_CLASSES - 1, dtype=int)     # default: "other"
    for i, (_, members) in enumerate(_ATOM_CLASSES):
        idx[np.isin(z, list(members))] = i
    return idx


def contact_feature_names() -> list[str]:
    labels = [name for name, _ in _ATOM_CLASSES] + ["other"]
    out = []
    for lo, hi in zip((0.0,) + CONTACT_SHELLS[:-1], CONTACT_SHELLS):
        for a in labels:
            for b in labels:
                out.append(f"contact_{a}{b}_{lo:g}-{hi:g}A")
    return out


def contact_features(
    atomic_nums: Sequence[int],
    positions: Sequence[tuple[float, float, float]],
    n_ligand: int,
    normalise: bool = True,
) -> np.ndarray:
    """Count ligand-protein atom pairs by element class and distance shell.

    Returns ``CONTACT_FEAT_DIM`` counts: one per (ligand class, protein class,
    shell). Hydrogens are ignored, matching the Vina terms, since protonation in
    the source structures is partial and inconsistent.

    With ``normalise`` the counts are divided by the ligand's heavy-atom count,
    so the features describe contacts *per ligand atom* rather than growing with
    molecular size. Raw counts correlate strongly with how many atoms are present,
    which is the shortcut that let an atom inventory stand in for the geometry.
    """
    z = np.asarray(atomic_nums, dtype=int)
    pos = np.asarray(positions, dtype=float)
    if z.shape[0] != pos.shape[0]:
        raise ValueError("atomic_nums and positions must have equal length")

    heavy = z != 1
    if not heavy.all():
        n_ligand = int(heavy[:n_ligand].sum())
        z, pos = z[heavy], pos[heavy]

    out = np.zeros(CONTACT_FEAT_DIM, dtype=float)
    if n_ligand <= 0 or n_ligand >= len(z):
        return out

    lig, poc = slice(0, n_ligand), slice(n_ligand, None)
    dist = np.linalg.norm(pos[lig][:, None, :] - pos[poc][None, :, :], axis=-1)
    lig_cls = _class_index(z[lig])
    poc_cls = _class_index(z[poc])
    pair_cls = lig_cls[:, None] * _N_CLASSES + poc_cls[None, :]

    lower = 0.0
    per_shell = _N_CLASSES * _N_CLASSES
    for s, upper in enumerate(CONTACT_SHELLS):
        in_shell = (dist > lower) & (dist <= upper)
        if in_shell.any():
            counts = np.bincount(pair_cls[in_shell], minlength=per_shell)
            out[s * per_shell:(s + 1) * per_shell] = counts
        lower = upper

    if normalise:
        out /= max(n_ligand, 1)
    return out
