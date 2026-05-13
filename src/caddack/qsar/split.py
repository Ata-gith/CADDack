from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as exc:
        raise ImportError(
            "RDKit is required for scaffold splitting. "
            "Install with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem, MurckoScaffold


def murcko_scaffold(smiles: str) -> Optional[str]:
    """Return canonical Murcko scaffold SMILES, or None if the input is invalid."""
    Chem, MurckoScaffold = _require_rdkit()
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(m)
    if core is None:
        return None
    return Chem.MolToSmiles(core, canonical=True)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES_canonical",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scaffolds: Dict[str, List[int]] = {}
    invalid: List[int] = []

    for i, s in enumerate(df[smiles_col].fillna("")):
        scaf = murcko_scaffold(s)
        if scaf is None:
            invalid.append(i)
        else:
            scaffolds.setdefault(scaf, []).append(i)

    keys = list(scaffolds.keys())
    rng.shuffle(keys)

    # invalid rows always go to train to avoid leaking bad data into evaluation
    test_idx: List[int] = []
    n_target = int(round(len(df) * test_size))
    for k in keys:
        test_idx.extend(scaffolds[k])
        if len(test_idx) >= n_target:
            break

    test_set = np.array(sorted(set(test_idx)), dtype=int)
    mask = np.ones(len(df), dtype=bool)
    mask[test_set] = False
    train_idx = np.where(mask)[0]
    return train_idx, test_set
