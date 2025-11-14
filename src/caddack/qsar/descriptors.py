from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np

try:
    from rdkit.Chem import rdFingerprintGenerator as rfg
    _HAS_GEN = True
except Exception:
    _HAS_GEN = False


# Physicochemical baseline
BASIC_DESC = [
    ("MolWt", Descriptors.MolWt),
    ("LogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHBD", rdMolDescriptors.CalcNumHBD),
    ("NumHBA", rdMolDescriptors.CalcNumHBA),
    ("NumRotBonds", rdMolDescriptors.CalcNumRotatableBonds),
]


def parse_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    m = parse_smiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def strip_salts(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    if "." not in smiles:
        return smiles
    best = None
    best_atoms = -1
    for frag in smiles.split("."):
        mol = parse_smiles(frag)
        if mol is None:
            continue
        n = mol.GetNumAtoms()
        if n > best_atoms:
            best, best_atoms = frag, n
    return best


def smiles_to_mol_clean(smiles: str) -> Chem.Mol | None:
    s = strip_salts(smiles)
    if s is None:
        return None
    s = canonicalize_smiles(s)
    if s is None:
        return None
    return parse_smiles(s)


def mol_to_basic_features(m: Chem.Mol) -> dict[str, float | int]:
    return {name: float(fn(m)) for name, fn in BASIC_DESC}


def mol_to_ecfp_bits(
    m: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, int]:
    """
    Return dense 0/1 dict for ECFP bits, compatible with multiple RDKit APIs.
    """
    if _HAS_GEN:
        # build generator (signature differences across RDKit versions)
        try:
            gen = rfg.GetMorganGenerator(
                radius=radius,
                fpSize=n_bits,
                includeChirality=True,
            )
        except TypeError:
            gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)

        # try common output methods in order
        if hasattr(gen, "GetFingerprintAsBitVect"):
            bv = gen.GetFingerprintAsBitVect(m)
            on = set(bv.GetOnBits())
        elif hasattr(gen, "GetCountFingerprint"):
            siv = gen.GetCountFingerprint(m)  # SparseIntVect
            on = set(siv.GetNonzeroElements().keys())
        elif hasattr(gen, "GetFingerprintAsNumPy"):
            arr = gen.GetFingerprintAsNumPy(m)
            on = set(np.nonzero(arr)[0].tolist())
        elif hasattr(gen, "GetFingerprint"):
            # some versions return ExplicitBitVect here
            fp = gen.GetFingerprint(m)
            try:
                on = set(fp.GetOnBits())
            except AttributeError:
                # SparseIntVect fallback
                on = set(fp.GetNonzeroElements().keys())
        else:
            on = set()
        return {f"ECFP{radius}_{i}": int(i in on) for i in range(n_bits)}

    # legacy fallback
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        m,
        radius=radius,
        nBits=n_bits,
    )
    return {f"ECFP{radius}_{i}": int(fp.GetBit(i)) for i in range(n_bits)}


def smiles_to_features(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict | None:
    m = smiles_to_mol_clean(smiles)
    if m is None:
        return None
    feats = mol_to_basic_features(m)
    feats.update(mol_to_ecfp_bits(m, radius=radius, n_bits=n_bits))
    feats["SMILES_canonical"] = Chem.MolToSmiles(m, canonical=True)
    return feats


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    radius: int = 2,
    n_bits: int = 2048,
    target_col: str | None = "pIC50",
    drop_na_target: bool = True,
) -> pd.DataFrame:
    """
    Featurize molecules and keep label columns (e.g. pIC50) as-is.

    - Expects df to already contain pIC50 from upstream fetch.
    - Coerces target_col to numeric.
    - Optionally drops rows with NaN target.
    """
    out_rows: list[dict] = []

    for _, row in df.iterrows():
        smi = row.get(smiles_col)
        f = smiles_to_features(smi, radius=radius, n_bits=n_bits)

        base = row.to_dict()

        if f is None:
            out_rows.append({**base, "__error": "invalid_smiles"})
        else:
            out_rows.append({**base, **f})

    out = pd.DataFrame(out_rows)

    if target_col is not None and target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
        if drop_na_target:
            out = out.loc[out[target_col].notna()].reset_index(drop=True)

    return out
