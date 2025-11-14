from __future__ import annotations
from typing import Iterable, Tuple, Dict, List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import pandas as pd

def murcko_scaffold(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core, canonical=True) if core is not None else ""

def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "SMILES_canonical",
    test_size: float = 0.2,
    seed: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    scaffolds: Dict[str, List[int]] = {}
    for i, s in enumerate(df[smiles_col].fillna("")):
        scaf = murcko_scaffold(s)
        scaffolds.setdefault(scaf, []).append(i)
    keys = list(scaffolds.keys())
    rng.shuffle(keys)
    test_idx: List[int] = []
    n_target = int(round(len(df) * test_size))
    for k in keys:
        test_idx.extend(scaffolds[k])
        if len(test_idx) >= n_target:
            break
    test_idx = np.array(sorted(set(test_idx)), dtype=int)
    mask = np.ones(len(df), dtype=bool)
    mask[test_idx] = False
    train_idx = np.where(mask)[0]
    return train_idx, test_idx
