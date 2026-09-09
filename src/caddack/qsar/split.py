"""Scaffold-based train/test splitting.

A scaffold split holds out whole Bemis-Murcko frameworks, so test compounds are
chemically novel relative to training ones. It is markedly harder than a random
split and is the honest way to estimate prospective performance.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:
        raise ImportError(
            "RDKit is required for scaffold splitting. "
            "Install with `pip install rdkit` (or `pip install caddack[gnn]`)."
        ) from exc
    return Chem, MurckoScaffold


def murcko_scaffold(smiles: str) -> str | None:
    """Return canonical Bemis-Murcko scaffold SMILES.

    Returns ``None`` for an invalid SMILES *and* for an acyclic molecule. RDKit
    returns an empty molecule for a ring-free input, whose SMILES is ``""``;
    treating that as a scaffold key would collapse every acyclic compound in a
    dataset into one group, which is chemically meaningless (Bemis & Murcko,
    J. Med. Chem. 39:2887, 1996 -- a framework is defined by its ring systems).
    Callers route ``None`` to the training set.
    """
    Chem, MurckoScaffold = _require_rdkit()
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(m)
    if core is None or core.GetNumAtoms() == 0:
        return None  # acyclic: no ring system, therefore no scaffold
    scaf = Chem.MolToSmiles(core, canonical=True)
    return scaf or None


@dataclass
class SplitReport:
    """What a scaffold split actually did, as opposed to what was requested."""

    requested_test_size: float
    achieved_test_size: float
    n_total: int
    n_train: int
    n_test: int
    n_scaffold_groups: int
    n_acyclic: int = 0
    n_invalid: int = 0
    largest_group: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "requested_test_size": self.requested_test_size,
            "achieved_test_size": self.achieved_test_size,
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_scaffold_groups": self.n_scaffold_groups,
            "n_acyclic": self.n_acyclic,
            "n_invalid": self.n_invalid,
            "largest_group": self.largest_group,
            "notes": list(self.notes),
        }


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES_canonical",
    test_size: float = 0.2,
    seed: int = 42,
    *,
    max_overshoot: float = 0.05,
    return_report: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, SplitReport]:
    """Split by Bemis-Murcko scaffold, holding out whole scaffold groups.

    Returns **positional** indices, for use with ``.iloc`` -- never ``.loc``.
    A frame whose index is not a plain ``RangeIndex`` is reset internally, so the
    returned integers always refer to row positions in the frame as passed.

    Groups are filled smallest-first and the loop stops *before* exceeding the
    target, so the achieved test fraction stays close to ``test_size`` instead of
    overshooting by the size of whichever large group happened to be drawn first.
    If the target still cannot be met within ``max_overshoot`` -- which happens
    when one scaffold dominates the dataset -- a ``RuntimeWarning`` reports the
    fraction actually achieved rather than silently returning it.

    Compounds with no scaffold (acyclic) and unparseable SMILES go to train, so
    neither can leak into the evaluation set.

    With ``return_report=True`` a third :class:`SplitReport` is returned, which is
    what benchmark scripts should persist alongside their metrics.
    """
    if not df.index.equals(pd.RangeIndex(len(df))):
        # Positional indices against a non-positional frame would be silently
        # wrong under .loc; normalise so the contract always holds.
        df = df.reset_index(drop=True)

    scaffolds: dict[str, list[int]] = {}
    acyclic: list[int] = []
    invalid: list[int] = []

    for i, s in enumerate(df[smiles_col].fillna("")):
        if not s:
            invalid.append(i)
            continue
        scaf = murcko_scaffold(s)
        if scaf is None:
            # Either unparseable or genuinely acyclic; distinguish for the report.
            from caddack.qsar.descriptors import parse_smiles

            (acyclic if parse_smiles(s) is not None else invalid).append(i)
        else:
            scaffolds.setdefault(scaf, []).append(i)

    n_total = len(df)
    n_target = int(round(n_total * test_size))

    # Deterministic ordering by size, then a seeded shuffle within each size band
    # so equal-sized groups are not always taken in dictionary order.
    rng = np.random.default_rng(seed)
    groups = sorted(scaffolds.values(), key=len)
    sizes = sorted({len(g) for g in groups})
    ordered: list[list[int]] = []
    for size in sizes:
        band = [g for g in groups if len(g) == size]
        rng.shuffle(band)
        ordered.extend(band)

    # Fill test from the smallest groups up, never crossing the overshoot bound.
    limit = n_target * (1.0 + max_overshoot)
    test_idx: list[int] = []
    for g in ordered:
        if len(test_idx) >= n_target:
            break
        if len(test_idx) + len(g) > limit:
            continue  # this group would overshoot; try a smaller one
        test_idx.extend(g)

    test_set = np.array(sorted(set(test_idx)), dtype=int)
    mask = np.ones(n_total, dtype=bool)
    mask[test_set] = False
    train_idx = np.where(mask)[0]

    achieved = len(test_set) / n_total if n_total else 0.0
    report = SplitReport(
        requested_test_size=float(test_size),
        achieved_test_size=float(achieved),
        n_total=n_total,
        n_train=int(len(train_idx)),
        n_test=int(len(test_set)),
        n_scaffold_groups=len(scaffolds),
        n_acyclic=len(acyclic),
        n_invalid=len(invalid),
        largest_group=max((len(g) for g in groups), default=0),
    )
    if abs(achieved - test_size) > max_overshoot:
        msg = (
            f"scaffold split achieved test fraction {achieved:.3f} "
            f"(requested {test_size:.3f}) from {len(scaffolds)} scaffold groups; "
            f"largest group holds {report.largest_group} of {n_total} compounds"
        )
        report.notes.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if return_report:
        return train_idx, test_set, report
    return train_idx, test_set
