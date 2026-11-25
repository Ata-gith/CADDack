#!/usr/bin/env python3
"""Check that the browser's feature computation matches Python's.

``docs/dock.js`` reimplements :mod:`caddack.docking.scoring` in JavaScript so a
docked pose can be scored client-side. A silent divergence would produce
confident, wrong affinities with nothing to indicate it, which is exactly what
happened on the similarity page when RDKit.js quietly ignored a chirality flag.
So the two implementations are compared numerically on real complexes.

    python scripts/check_browser_features.py --data-dir /path/to/pdbbind

Writes PDBQT fixtures from PDBbind geometry, computes the expected features in
Python, runs the JavaScript through headless Chromium, and reports the largest
disagreement per feature group.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from caddack.affinity.features import feature_names  # noqa: E402
from caddack.docking.scoring import (  # noqa: E402
    TERM_NAMES,
    contact_features,
    vina_terms,
)
from caddack.qsar.descriptors import (  # noqa: E402
    mol_to_basic_features,
    mol_to_ecfp_array,
    smiles_to_mol_clean,
)

Z_SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
            15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 25: "Mn", 26: "Fe",
            29: "Cu", 30: "Zn", 34: "Se", 35: "Br", 53: "I"}
# AutoDock types the browser has to map back to an element, including the
# suffixed ones that carry hydrogen-bonding information.
AD_TYPE = {6: "C", 7: "NA", 8: "OA", 16: "SA", 1: "HD"}


def write_pdbqt(atomic_nums, positions) -> str:
    """A minimal but column-correct PDBQT, as Vina and Meeko emit."""
    lines = []
    for i, (z, (x, y, zc)) in enumerate(zip(atomic_nums, positions), start=1):
        element = Z_SYMBOL.get(int(z), "C")
        ad = AD_TYPE.get(int(z), element.upper()[:2])
        name = f"{element}{i}"[:4]
        # Explicit PDB columns. Getting these wrong shifts the coordinates by a
        # column and the parser silently reads the wrong field, which is what the
        # first version of this writer did.
        #   1-6 record | 7-11 serial | 13-16 name | 17 altLoc | 18-20 resName
        #   22 chain | 23-26 resSeq | 31-38 x | 39-46 y | 47-54 z
        #   55-60 occupancy | 61-66 temp | 71-76 charge | 78-79 AutoDock type
        lines.append(
            f"ATOM  {i:>5} {name:<4}{' '}{'LIG':>3} {'A'}{1:>4}{' '}   "
            f"{x:8.3f}{y:8.3f}{zc:8.3f}{1.0:6.2f}{0.0:6.2f}    {0.0:6.3f} {ad:<2}"
        )
    return "\n".join(lines) + "\nTER\n"


def python_features(z, pos, n_ligand, smiles, fp_bits):
    mol = smiles_to_mol_clean(smiles)
    desc = mol_to_basic_features(mol)
    terms = vina_terms(z, pos, n_ligand=n_ligand)
    contacts = contact_features(z, pos, n_ligand=n_ligand)
    blocks = [
        [desc[k] for k in ("MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds")],
        [mol.GetNumAtoms()],
        [terms[k] for k in TERM_NAMES],
        [terms["vina_inter"]],
        contacts,
    ]
    if fp_bits:
        blocks.append(np.unpackbits(mol_to_ecfp_array(mol, radius=2, n_bits=fp_bits)))
    return np.concatenate(blocks), desc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--n-cases", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--fingerprint-bits", type=int, default=None,
                    help="ECFP4 bits to check. Defaults to whatever "
                         "docs/affinity.json says the shipped model uses, so "
                         "the check follows the model rather than a constant.")
    args = ap.parse_args(argv)

    fp_bits = args.fingerprint_bits
    if fp_bits is None:
        bundle = json.loads((REPO_ROOT / "docs" / "affinity.json").read_text())
        fp_bits = int(bundle.get("fingerprint_bits", 0))
    print(f"  checking {fp_bits} fingerprint bits "
          f"({'from docs/affinity.json' if args.fingerprint_bits is None else 'requested'})")

    with open(args.data_dir / "pdbbind_examples.pkl", "rb") as fh:
        examples = [e for e in pickle.load(fh) if getattr(e, "is_refined", True)]

    cases, expected = [], []
    for ex in examples:
        if len(cases) >= args.n_cases:
            break
        if not ex.ligand_smiles or smiles_to_mol_clean(ex.ligand_smiles) is None:
            continue
        z = list(ex.geo.atomic_nums)
        pos = [tuple(p) for p in ex.geo.positions]
        n_lig = ex.geo.n_ligand
        vec, desc = python_features(z, pos, n_lig, ex.ligand_smiles, fp_bits)
        cases.append({
            "id": ex.pdb_id,
            "ligand": write_pdbqt(z[:n_lig], pos[:n_lig]),
            "receptor": write_pdbqt(z[n_lig:], pos[n_lig:]),
            "descriptors": {k: float(desc[k]) for k in
                            ("MolWt", "LogP", "TPSA", "NumHBD", "NumHBA", "NumRotBonds")},
            "smiles": ex.ligand_smiles,
        })
        expected.append(vec)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"cases": cases, "fp_bits": fp_bits}, fh)
        fixtures = fh.name

    proc = subprocess.run(
        ["node", str(REPO_ROOT / "scripts" / "check_browser_features.mjs"), fixtures],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        return 1
    got = json.loads(proc.stdout)["results"]

    names = feature_names(fp_bits)
    groups = {
        "descriptors (6)": range(0, 6),
        "ligand atom count": range(6, 7),
        "vina terms (6)": range(7, 13),
        "contact counts (100)": range(13, 113),
    }
    if fp_bits:
        groups[f"ECFP4 bits ({fp_bits})"] = range(113, 113 + fp_bits)
    print(f"  {len(got)} complexes, {len(names)} features each\n")
    print(f"  {'group':<24}{'largest |python - browser|':>28}")
    worst_overall = 0.0
    for label, idx in groups.items():
        worst = 0.0
        for exp, js in zip(expected, got):
            a = np.asarray(exp)[list(idx)]
            b = np.asarray(js["features"])[list(idx)]
            worst = max(worst, float(np.max(np.abs(a - b))))
        worst_overall = max(worst_overall, worst)
        print(f"  {label:<24}{worst:>28.3e}")

    # Parsing has to recover every atom, or the features describe a different complex.
    parse_ok = all(js["n_ligand_parsed"] + js["n_receptor_parsed"] ==
                   len(c["ligand"].strip().splitlines()) - 1 +
                   len(c["receptor"].strip().splitlines()) - 1
                   for c, js in zip(cases, got))
    print(f"\n  every atom recovered from the PDBQT: {parse_ok}")
    ok = worst_overall < args.tolerance and parse_ok
    print(f"  VERDICT: {'browser == python' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
