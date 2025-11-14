#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import csv
import requests
import math
import statistics as stats

# Endpoints
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_FILE_BASE = "https://files.rcsb.org/download"  # {PDB}.pdb or {PDB}.cif

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_chembl(chembl_id: str, smiles: str) -> str:
    p = REPO_ROOT / "data" / "raw" / "chembl" / f"{chembl_id}.smi"
    _write_text(p, smiles + "\n")
    return str(p)


def save_uniprot(uniprot_id: str, fasta: str) -> str:
    p = REPO_ROOT / "data" / "raw" / "uniprot" / f"{uniprot_id}.fasta"
    _write_text(p, fasta + ("\n" if not fasta.endswith("\n") else ""))
    return str(p)


UA = {"User-Agent": "CADDack/0.1 (+https://github.com/Ata-gith/CADDack)"}


def _get(url, *, timeout, stream=False, headers=None, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            return requests.get(
                url,
                timeout=timeout,
                stream=stream,
                headers={**(headers or {}), **UA},
            )
        except requests.RequestException as e:
            last = e
    raise last


def fetch_chembl_smiles(chembl_id: str, timeout: float = 20.0) -> dict:
    """Return: {"id": chembl_id, "smiles": <str|None>, "error": <str|None>}"""
    url = f"{CHEMBL_BASE}/molecule/{chembl_id}.json"
    try:
        r = _get(url, timeout=timeout)
        if r.status_code != 200:
            return {"id": chembl_id, "smiles": None, "error": f"http {r.status_code}"}
        data = r.json()
        smiles = (data.get("molecule_structures") or {}).get("canonical_smiles")
        if not smiles:
            return {"id": chembl_id, "smiles": None, "error": "no canonical_smiles"}
        return {"id": chembl_id, "smiles": smiles, "error": None}
    except requests.RequestException as e:
        return {"id": chembl_id, "smiles": None, "error": str(e)}


def _ic50_to_pIC50(value: float, units: str) -> float | None:
    """
    Convert IC50 value with given units to pIC50.

    Supported units:
      - nM: pIC50 = 9 - log10(IC50_nM)
      - uM: pIC50 = 6 - log10(IC50_uM)
      - M:  pIC50 = -log10(IC50_M)
    """
    if value <= 0:
        return None
    u = units.strip().upper()
    if u == "NM":
        return 9.0 - math.log10(value)
    if u in ("UM", "ΜM"):  # handle possible weird micro symbols
        return 6.0 - math.log10(value)
    if u == "M":
        return -math.log10(value)
    return None


def fetch_chembl_pIC50(chembl_id: str, timeout: float = 20.0) -> dict:
    """
    Fetch IC50 activities from ChEMBL and aggregate to a single pIC50.

    Return: {"id": chembl_id, "pIC50": <float|None>, "error": <str|None>}
    """
    # standard_type=IC50, limit large enough to catch all; rely on server defaults otherwise
    url = (
        f"{CHEMBL_BASE}/activity.json?"
        f"molecule_chembl_id={chembl_id}&standard_type=IC50&limit=1000"
    )
    try:
        r = _get(url, timeout=timeout)
        if r.status_code != 200:
            return {"id": chembl_id, "pIC50": None, "error": f"http {r.status_code}"}
        data = r.json()
        activities = data.get("activities") or []

        vals = []
        for a in activities:
            stype = (a.get("standard_type") or "").upper()
            if stype != "IC50":
                continue
            units = a.get("standard_units")
            val = a.get("standard_value")
            rel = (a.get("relation") or "=").strip()
            # use only ~ and = for a cleaner point estimate
            if rel not in ("=", "~"):
                continue
            if units is None or val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            p = _ic50_to_pIC50(v, units)
            if p is not None and math.isfinite(p):
                vals.append(p)

        if not vals:
            return {"id": chembl_id, "pIC50": None, "error": "no_usable_ic50"}

        pic50 = float(stats.median(vals))
        return {"id": chembl_id, "pIC50": pic50, "error": None}
    except requests.RequestException as e:
        return {"id": chembl_id, "pIC50": None, "error": str(e)}


def fetch_uniprot_fasta(uniprot_id: str, timeout: float = 20.0) -> dict:
    """Return: {"id": uniprot_id, "fasta": <str|None>, "error": <str|None>}"""
    url = f"{UNIPROT_BASE}/{uniprot_id}.fasta"
    try:
        r = _get(url, timeout=timeout, headers={"Accept": "text/x-fasta"})
        if r.status_code != 200:
            return {"id": uniprot_id, "fasta": None, "error": f"http {r.status_code}"}
        text = r.text.strip()
        if not text.startswith(">"):
            return {"id": uniprot_id, "fasta": None, "error": "invalid fasta payload"}
        return {"id": uniprot_id, "fasta": text, "error": None}
    except requests.RequestException as e:
        return {"id": uniprot_id, "fasta": None, "error": str(e)}


def fetch_pdb_file(
    pdb_id: str,
    out_dir: str | None = None,
    fmt: str = "pdb",
    timeout: float = 30.0,
) -> dict:
    """Download PDB/mmCIF from RCSB into <repo>/data/raw/pdb by default.
    Return: {"id": pdb_id, "path": <str|None>, "error": <str|None>}
    """
    pdb_code = pdb_id.upper().strip()
    if fmt not in {"pdb", "cif"}:
        return {"id": pdb_id, "path": None, "error": "fmt must be 'pdb' or 'cif'"}

    base_out = (REPO_ROOT / "data" / "raw" / "pdb") if out_dir is None else Path(out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    out_path = base_out / f"{pdb_code}.{fmt}"
    url = f"{RCSB_FILE_BASE}/{pdb_code}.{fmt}"

    try:
        r = _get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return {"id": pdb_id, "path": None, "error": f"http {r.status_code}"}
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return {"id": pdb_id, "path": str(out_path), "error": None}
    except requests.RequestException as e:
        return {"id": pdb_id, "path": None, "error": str(e)}


def add_cli(subparsers):
    p = subparsers.add_parser(
        "fetch",
        help="Fetch structures from public databases (ChEMBL, UniProt, PDB)",
    )
    p.add_argument("--chembl", nargs="*", default=[], help="ChEMBL IDs")
    p.add_argument("--uniprot", nargs="*", default=[], help="UniProt accessions")
    p.add_argument("--pdb", nargs="*", default=[], help="PDB IDs")
    p.add_argument(
        "--pdb-fmt",
        choices=["pdb", "cif"],
        default="pdb",
        help="PDB download format",
    )
    p.add_argument(
        "--pdb-outdir",
        default=str(REPO_ROOT / "data" / "raw" / "pdb"),
        help="Directory to save PDB files",
    )
    p.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    p.add_argument(
        "--emit-mols-csv",
        default=None,
        help="If set, write a CSV of fetched ligands (SMILES,chembl_id,pIC50)",
    )
    p.set_defaults(func=run)


def run(args):
    result = {
        "chembl": {},
        "uniprot": {},
        "pdb": {},
        "meta": {"ok": True, "errors": []},
    }
    chembl_rows = []

    # ChEMBL
    for cid in args.chembl:
        rec_smiles = fetch_chembl_smiles(cid)
        if rec_smiles["smiles"]:
            path = save_chembl(cid, rec_smiles["smiles"])
            result["chembl"][cid] = {"smiles_path": path}
            # try to fetch pIC50
            rec_pic50 = fetch_chembl_pIC50(cid)
            if rec_pic50["error"]:
                result["meta"]["errors"].append(
                    {"chembl_pIC50": cid, "msg": rec_pic50["error"]}
                )
                pic50 = None
            else:
                pic50 = rec_pic50["pIC50"]
            chembl_rows.append(
                {"SMILES": rec_smiles["smiles"], "chembl_id": cid, "pIC50": pic50}
            )
        else:
            result["chembl"][cid] = None
            result["meta"]["errors"].append(
                {"chembl": cid, "msg": rec_smiles["error"]}
            )

    # UniProt
    for uid in args.uniprot:
        rec = fetch_uniprot_fasta(uid)
        if rec["fasta"]:
            path = save_uniprot(uid, rec["fasta"])
            result["uniprot"][uid] = {"fasta_path": path}
        else:
            result["uniprot"][uid] = None
            result["meta"]["errors"].append({"uniprot": uid, "msg": rec["error"]})

    # PDB
    for pid in args.pdb:
        rec = fetch_pdb_file(pid, out_dir=args.pdb_outdir, fmt=args.pdb_fmt)
        if rec["path"]:
            result["pdb"][pid] = {"path": rec["path"]}
        else:
            result["pdb"][pid] = None
            result["meta"]["errors"].append({"pdb": pid, "msg": rec["error"]})

    if result["meta"]["errors"]:
        result["meta"]["ok"] = False

    payload = json.dumps(result, indent=2)
    if args.out == "-" or args.out.strip() == "":
        sys.stdout.write(payload + "\n")
    else:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(payload, encoding="utf-8")

    if args.emit_mols_csv:
        out_csv = Path(args.emit_mols_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["SMILES", "chembl_id", "pIC50"]
            )
            w.writeheader()
            w.writerows(chembl_rows)

    if result["meta"]["errors"]:
        result["meta"]["ok"] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
