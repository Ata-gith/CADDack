from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except Exception as exc:
        raise ImportError(
            "RDKit is required for QSAR descriptors. "
            "Install with `pip install rdkit` (or `pip install caddack[gnn]`)."
        ) from exc
    return Chem, Descriptors, rdMolDescriptors


def _get_rfg():
    """Return rdFingerprintGenerator module or None if unavailable."""
    try:
        from rdkit.Chem import rdFingerprintGenerator as rfg
        return rfg
    except Exception:
        return None


def _basic_desc():
    """Return physicochemical descriptor list, resolving RDKit callables lazily."""
    _, Descriptors, rdMolDescriptors = _require_rdkit()
    return [
        ("MolWt", Descriptors.MolWt),
        ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("NumHBD", rdMolDescriptors.CalcNumHBD),
        ("NumHBA", rdMolDescriptors.CalcNumHBA),
        ("NumRotBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ]


def parse_smiles(smiles: str) -> Any:
    if not isinstance(smiles, str):
        return None
    Chem, _, _ = _require_rdkit()
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    m = parse_smiles(smiles)
    if m is None:
        return None
    Chem, _, _ = _require_rdkit()
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


def smiles_to_mol_clean(smiles: str, standardize: bool = False) -> Any:
    """SMILES -> cleaned Mol, parsing once rather than three times.

    The previous implementation parsed each salt fragment, then re-parsed to
    canonicalise, then parsed again -- three to four RDKit parses per molecule.

    ``standardize=True`` routes through the full ChEMBL-style pipeline
    (neutralisation, functional-group normalisation) instead of largest-fragment
    selection alone. See :mod:`caddack.qsar.standardize`.
    """
    if standardize:
        from caddack.qsar.standardize import standardize_mol

        mol = parse_smiles(smiles)
        return standardize_mol(mol) if mol is not None else None

    s = strip_salts(smiles)
    if s is None:
        return None
    return parse_smiles(s)   # already the largest fragment; canonicalise on output


def mol_to_basic_features(m: Any) -> dict[str, float | int]:
    return {name: float(fn(m)) for name, fn in _basic_desc()}


def mol_to_ecfp_bits(
    m: Any,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, int]:
    """Return dense 0/1 dict for ECFP bits, compatible with multiple RDKit APIs.

    Column names follow the ECFP convention, in which the number is the circular
    **diameter**, not the radius (Rogers & Hahn, JCIM 50:742, 2010). The default
    ``radius=2`` therefore yields ``ECFP4_*`` columns, not ``ECFP2_*``.
    """
    rfg = _get_rfg()
    if rfg is not None:
        try:
            gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        except TypeError:
            gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)

        if hasattr(gen, "GetFingerprintAsBitVect"):
            bv = gen.GetFingerprintAsBitVect(m)
            on = set(bv.GetOnBits())
        elif hasattr(gen, "GetCountFingerprint"):
            siv = gen.GetCountFingerprint(m)
            on = set(siv.GetNonzeroElements().keys())
        elif hasattr(gen, "GetFingerprintAsNumPy"):
            arr = gen.GetFingerprintAsNumPy(m)
            on = set(np.nonzero(arr)[0].tolist())
        elif hasattr(gen, "GetFingerprint"):
            fp = gen.GetFingerprint(m)
            try:
                on = set(fp.GetOnBits())
            except AttributeError:
                on = set(fp.GetNonzeroElements().keys())
        else:
            on = set()
        return {f"ECFP{2 * radius}_{i}": int(i in on) for i in range(n_bits)}

    # legacy fallback — useChirality=True to match the generator path above,
    # so bit meanings are identical regardless of which RDKit API is present
    _, _, rdMolDescriptors = _require_rdkit()
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        m, radius=radius, nBits=n_bits, useChirality=True
    )
    return {f"ECFP{2 * radius}_{i}": int(fp.GetBit(i)) for i in range(n_bits)}


def mol_to_ecfp_array(m: Any, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP as a bit-packed uint8 array of length ``n_bits // 8``.

    The dense dict form costs 8 bytes per bit (int64), so 2048 bits is 16 KB per
    molecule against 256 bytes packed -- a factor of 64. At 1e5 molecules that is
    1.6 GB versus 26 MB. Use this for anything at scale; unpack with
    ``np.unpackbits`` when individual bits are needed, and compute Tanimoto by
    popcount directly on the packed form.
    """
    rfg = _get_rfg()
    if rfg is not None:
        try:
            gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        except TypeError:
            gen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
        if hasattr(gen, "GetFingerprintAsNumPy"):
            return np.packbits(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
    bits = mol_to_ecfp_bits(m, radius=radius, n_bits=n_bits)
    dense = np.fromiter((bits[f"ECFP{2 * radius}_{i}"] for i in range(n_bits)),
                        dtype=np.uint8, count=n_bits)
    return np.packbits(dense)


def smiles_to_features(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    standardize: bool = False,
) -> dict | None:
    m = smiles_to_mol_clean(smiles, standardize=standardize)
    if m is None:
        return None
    Chem, _, _ = _require_rdkit()
    feats = mol_to_basic_features(m)
    feats.update(mol_to_ecfp_bits(m, radius=radius, n_bits=n_bits))
    feats["SMILES_canonical"] = Chem.MolToSmiles(m, canonical=True)
    return feats


def _featurize_one(smiles, radius: int = 2, n_bits: int = 2048,
                   standardize: bool = False) -> dict | None:
    """Featurise one SMILES. Module level so ProcessPoolExecutor can pickle it."""
    return smiles_to_features(smiles, radius=radius, n_bits=n_bits,
                              standardize=standardize)


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    radius: int = 2,
    n_bits: int = 2048,
    target_col: str | None = "pIC50",
    drop_na_target: bool = True,
    standardize: bool = False,
    n_jobs: int = 1,
    chunksize: int = 256,
) -> pd.DataFrame:
    """Featurise molecules, keeping label columns (e.g. pIC50) as they are.

    - Expects ``df`` to already carry the target from the upstream fetch.
    - Coerces ``target_col`` to numeric and optionally drops NaN targets.
    - ``standardize=True`` applies the ChEMBL-style curation pipeline
      (see :mod:`caddack.qsar.standardize`) instead of salt stripping alone.

    ``n_jobs`` parallelises across processes. It is 1 by default because process
    startup dominates on small inputs -- measured on 4 cores, 200 molecules ran
    0.6x (i.e. slower), 1,000 ran 1.35x and 4,000 ran 2.2x. Turn it on above
    roughly a thousand molecules.
    """
    smiles_list = df[smiles_col].tolist()
    base_rows = df.to_dict("records")   # one pass, instead of a Series per row

    if n_jobs and n_jobs != 1 and len(smiles_list) > 1:
        # Ship SMILES strings to the workers, not Mol objects: RDKit molecules
        # pickle expensively, so parsing inside the worker is the cheaper split.
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        worker = partial(_featurize_one, radius=radius, n_bits=n_bits,
                         standardize=standardize)
        max_workers = None if n_jobs in (-1, 0) else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            feats = list(pool.map(worker, smiles_list, chunksize=chunksize))
    else:
        feats = [_featurize_one(s, radius=radius, n_bits=n_bits,
                                standardize=standardize) for s in smiles_list]

    out_rows = [
        {**base, "__error": "invalid_smiles"} if f is None else {**base, **f}
        for base, f in zip(base_rows, feats)
    ]
    out = pd.DataFrame(out_rows)

    if target_col is not None and target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
        if drop_na_target:
            out = out.loc[out[target_col].notna()].reset_index(drop=True)

    return out
