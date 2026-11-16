"""``caddack dock`` -- run AutoDock Vina from the command line.

The docking module had no CLI entry point, so the only way to reach 189 lines of
working code was to import it from Python. This exposes it as a subcommand, in
the same shape as the other subcommands: the logic lives in the package and
``scripts/run_vina.py`` is a thin wrapper.

Every heavy import is deferred into :func:`run`, so ``caddack --help`` works
without RDKit, Meeko or Vina installed.
"""
from __future__ import annotations

import json
from pathlib import Path


def add_cli(subparsers):
    p = subparsers.add_parser(
        "dock", help="Dock a SMILES ligand into a receptor PDB with AutoDock Vina")
    p.add_argument("--receptor", required=True,
                   help="Receptor PDB. Give a COMPLETE protein, not a truncated "
                        "pocket: Meeko's residue templates need a well-formed "
                        "polymer, and the fallback writer loses polar hydrogens")
    p.add_argument("--smiles", required=True, help="Ligand SMILES to dock")

    box = p.add_argument_group(
        "search box", "Give either --reference-ligand or both --center and --size")
    box.add_argument("--reference-ligand", default=None,
                     help="Co-crystallised ligand (.sdf/.mol2) to centre the box on")
    box.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     default=None, help="Box centre in angstroms")
    box.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     default=None, help="Box size in angstroms")

    p.add_argument("--exhaustiveness", type=int, default=8,
                   help="Vina search effort (default 8). 32 is measurably more "
                        "reliable at roughly four times the runtime")
    p.add_argument("--n-poses", type=int, default=9, help="Poses to return")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-hetatm", nargs="*", default=None, metavar="RESNAME",
                   help="HETATM residues to keep in the receptor, e.g. ZN HEM. "
                        "Everything else (waters, additives, the co-crystallised "
                        "ligand) is stripped")
    p.add_argument("--workdir", default=None,
                   help="Where intermediate PDBQT files go (default: a temp dir)")
    p.add_argument("--out", default=None, help="Write results as JSON to this path")
    p.add_argument("--save-poses", default=None, metavar="DIR",
                   help="Write each pose as a .pdbqt file into this directory")
    p.add_argument("--rmsd-to", default=None, metavar="LIGAND",
                   help="Reference ligand (.sdf/.mol2) to score poses against. "
                        "Below 2 A is conventionally counted as reproducing the "
                        "crystal pose")
    p.set_defaults(func=run)


def _load_reference_mol(path: str | Path):
    """Load a reference ligand for RMSD comparison."""
    from rdkit import Chem

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        mol = next((m for m in Chem.SDMolSupplier(str(path), removeHs=True)
                    if m is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=True)
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=True)
    else:
        raise SystemExit(f"unsupported reference ligand format: {suffix!r}")
    if mol is None:
        raise SystemExit(f"could not read a molecule from {path}")
    return mol


def _resolve_box(args):
    from caddack.docking import Box, box_from_reference_ligand

    if args.reference_ligand:
        if args.center or args.size:
            raise SystemExit("give either --reference-ligand or --center/--size, not both")
        return box_from_reference_ligand(args.reference_ligand)
    if args.center and args.size:
        return Box(center=tuple(args.center), size=tuple(args.size))
    raise SystemExit(
        "no search box: give --reference-ligand, or both --center and --size"
    )


def run(args):
    from caddack.docking import dock_smiles, pose_rmsd

    box = _resolve_box(args)
    poses = dock_smiles(
        args.receptor,
        args.smiles,
        box=box,
        workdir=args.workdir,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        keep_hetatm=args.keep_hetatm,
    )
    if not poses:
        raise SystemExit("Vina returned no poses; check the box covers the site")

    reference = _load_reference_mol(args.rmsd_to) if args.rmsd_to else None

    rows = []
    for pose in poses:
        row = {"rank": pose.rank, "score": pose.score,
               "rmsd_lb": pose.rmsd_lb, "rmsd_ub": pose.rmsd_ub}
        if reference is not None:
            try:
                row["rmsd_to_reference"] = pose_rmsd(pose, reference)
            except Exception as exc:
                # A pose that cannot be matched to the reference is reported as
                # such rather than silently dropped or scored as zero.
                row["rmsd_to_reference"] = None
                row["rmsd_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    if args.save_poses:
        outdir = Path(args.save_poses)
        outdir.mkdir(parents=True, exist_ok=True)
        for pose in poses:
            (outdir / f"pose_{pose.rank:02d}.pdbqt").write_text(
                pose.pdbqt, encoding="utf-8")

    result = {
        "receptor": args.receptor,
        "smiles": args.smiles,
        "box": {"center": list(box.center), "size": list(box.size)},
        "exhaustiveness": args.exhaustiveness,
        "seed": args.seed,
        "poses": rows,
        "best_score": rows[0]["score"],
    }
    if reference is not None:
        valid = [r["rmsd_to_reference"] for r in rows
                 if r.get("rmsd_to_reference") is not None]
        result["top_pose_rmsd"] = rows[0].get("rmsd_to_reference")
        result["best_rmsd_any_pose"] = min(valid) if valid else None

    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    print(payload)
