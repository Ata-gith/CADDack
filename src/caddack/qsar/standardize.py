"""Chemical structure standardisation.

Two molecules that are the same compound drawn differently -- a hydrochloride
salt versus its free base, one tautomer versus another, a nitro group written
two ways -- produce different canonical SMILES, hence different fingerprints,
different scaffold groups, and duplicate rows that can land on both sides of a
train/test split. Standardising first is what makes deduplication and splitting
mean what they claim to.

The operation order follows the ChEMBL curation pipeline (Bento et al.,
"An open source chemical structure curation pipeline using RDKit",
J. Cheminform. 12:51, 2020):

    cleanup -> largest organic fragment -> neutralise -> normalise -> (tautomer)

Tautomer canonicalisation is separate and opt-in because it is markedly slower.
"""
from __future__ import annotations

from functools import lru_cache


def _require_standardizer():
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "RDKit is required for structure standardisation. "
            "Install with `pip install rdkit` (or `pip install caddack[gnn]`)."
        ) from exc
    return Chem, rdMolStandardize


@lru_cache(maxsize=1)
def _tools():
    """Build the standardiser objects once; they are stateless and reusable."""
    _, rdMolStandardize = _require_standardizer()
    return (
        rdMolStandardize.LargestFragmentChooser(),
        rdMolStandardize.Uncharger(),
        rdMolStandardize.TautomerEnumerator(),
    )


def standardize_mol(mol, *, neutralize: bool = True, canonical_tautomer: bool = False):
    """Standardise an RDKit ``Mol``. Returns ``None`` if the molecule is unusable.

    ``neutralize`` strips counter-ion charges left behind after salt removal --
    without it a stripped hydrochloride keeps its ``[NH3+]``, which changes both
    descriptors and fingerprint bits relative to the neutral form.

    ``canonical_tautomer`` additionally maps the molecule to a canonical tautomer.
    It is off by default because enumeration is expensive on large molecules; turn
    it on when deduplicating or splitting, where two tautomers of one compound
    landing in different groups is a genuine leakage path.
    """
    Chem, rdMolStandardize = _require_standardizer()
    if mol is None:
        return None
    chooser, uncharger, tautomers = _tools()
    try:
        mol = rdMolStandardize.Cleanup(mol)      # sanitise, normalise, disconnect metals
        if mol is None:
            return None
        mol = chooser.choose(mol)                # largest organic fragment
        if neutralize:
            mol = uncharger.uncharge(mol)
        mol = rdMolStandardize.Normalize(mol)    # canonical functional-group forms
        if canonical_tautomer:
            mol = tautomers.Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def standardize_smiles(
    smiles: str,
    *,
    neutralize: bool = True,
    canonical_tautomer: bool = False,
) -> str | None:
    """Standardise a SMILES string, returning canonical SMILES or ``None``."""
    Chem, _ = _require_standardizer()
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(mol, neutralize=neutralize,
                          canonical_tautomer=canonical_tautomer)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def inchikey(smiles: str) -> str | None:
    """InChIKey of a standardised structure -- the right key for deduplication.

    Cheaper and more reliable to compare than SMILES, and stable across the
    spellings that standardisation collapses.
    """
    Chem, _ = _require_standardizer()
    std = standardize_smiles(smiles)
    if std is None:
        return None
    mol = Chem.MolFromSmiles(std)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None
