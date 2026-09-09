#!/usr/bin/env python3
"""Re-docking validation for the docking pipeline.

Docks each PDBbind ligand back into its own receptor and measures RMSD to the
crystal pose. Reproducing the crystal pose within 2 A is the conventional
success criterion, so this answers "does our docking find the right answer?"
rather than merely "does it run?".

Assumes a PDBbind-style layout (as produced by ``benchmark_pdbbind.py --build``)::

    <data-dir>/v2015/<pdb_id>/<pdb_id>_protein.pdb
    <data-dir>/v2015/<pdb_id>/<pdb_id>_ligand.mol2

Usage:
    python scripts/benchmark_redocking.py --data-dir /tmp/pdbbind --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import warnings
from pathlib import Path


def _refined_ids(data_dir: Path) -> list[str]:
    index = data_dir / "v2015" / "INDEX_refined_data.2015"
    if not index.exists():
        raise SystemExit(f"{index} not found; run benchmark_pdbbind.py --download first")
    ids = []
    for line in index.read_text(errors="ignore").splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            ids.append(parts[0])
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=30, help="complexes to sample")
    ap.add_argument("--exhaustiveness", type=int, default=32,
                help="Vina search effort. Its own default is 8; 32 is worth "
                     "the extra time for benchmarking (+7 points here).")
    ap.add_argument("--n-poses", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="explicit PDB ids")
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--start", choices=["crystal", "embed"], default="crystal",
        help="starting ligand conformer. 'crystal' (default, and the standard "
             "re-docking protocol) keeps the deposited 3D structure; Vina then "
             "randomises position and orientation and samples acyclic torsions. "
             "'embed' regenerates the conformer with ETKDG, which also tests "
             "conformer generation — note Vina cannot change ring puckers, so a "
             "wrong ring conformation caps the achievable RMSD.")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    from caddack.docking import (
        box_from_reference_ligand,
        dock_pdbqt,
        ligand_pdbqt_from_mol,
        pose_rmsd,
        receptor_pdbqt_from_pdb,
    )

    data_dir = Path(args.data_dir)
    ids = args.ids or random.Random(args.seed).sample(_refined_ids(data_dir), args.n)
    work = data_dir / "_redock"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for pid in ids:
        d = data_dir / "v2015" / pid
        ligand, protein = d / f"{pid}_ligand.mol2", d / f"{pid}_protein.pdb"
        if not (ligand.exists() and protein.exists()):
            rows.append({"pdb": pid, "status": "missing_files"})
            continue
        ref = Chem.MolFromMol2File(str(ligand))
        if ref is None:
            rows.append({"pdb": pid, "status": "ligand_parse_failed"})
            continue
        try:
            box = box_from_reference_ligand(ligand)
            # Vina randomises translation, rotation and acyclic torsions itself,
            # so starting from the crystal conformer does not hand it the answer.
            # It does keep ring conformations from the input, which is why
            # re-embedding is a separate (harder) test rather than the default.
            probe = Chem.AddHs(Chem.Mol(ref), addCoords=True)
            if args.start == "embed":
                if AllChem.EmbedMolecule(probe, randomSeed=7) != 0:
                    rows.append({"pdb": pid, "status": "embed_failed"})
                    continue
            lig_q = ligand_pdbqt_from_mol(probe, work / f"{pid}_ligand.pdbqt")
            rec_q = receptor_pdbqt_from_pdb(protein, work / f"{pid}_receptor.pdbqt")
            poses = dock_pdbqt(rec_q, lig_q, box,
                               exhaustiveness=args.exhaustiveness,
                               n_poses=args.n_poses)
            top = pose_rmsd(poses[0], ref)
            best = min(pose_rmsd(p, ref) for p in poses)
            rows.append({"pdb": pid, "status": "ok", "score": poses[0].score,
                         "rmsd_top": top, "rmsd_best": best})
            print(f"  {pid}  score {poses[0].score:7.2f}  "
                  f"RMSD top {top:5.2f}  best {best:5.2f}", flush=True)
        except Exception as exc:
            rows.append({"pdb": pid, "status": f"error: {type(exc).__name__}"})
            print(f"  {pid}  FAILED {type(exc).__name__}: {exc}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\nDocked {len(ok)}/{len(rows)} complexes")
    if ok:
        top2 = sum(r["rmsd_top"] < 2.0 for r in ok)
        best2 = sum(r["rmsd_best"] < 2.0 for r in ok)
        print(f"  top pose      < 2 A : {top2}/{len(ok)} = {100*top2/len(ok):.0f}%")
        print(f"  any of top {args.n_poses} < 2 A : {best2}/{len(ok)} = {100*best2/len(ok):.0f}%")
        print(f"  median RMSD (top)   : {statistics.median(r['rmsd_top'] for r in ok):.2f} A")
        print(f"  median RMSD (best)  : {statistics.median(r['rmsd_best'] for r in ok):.2f} A")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
