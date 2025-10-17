#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import requests

# Endpoints
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_FILE_BASE = "https://files.rcsb.org/download"  # {PDB}.pdb or {PDB}.cif


def fetch_chembl_smiles(chembl_id: str, timeout: float = 20.0) -> dict:
    """Return: {"id": chembl_id, "smiles": <str|None>, "error": <str|None>}"""
    url = f"{CHEMBL_BASE}/molecule/{chembl_id}.json"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {"id": chembl_id, "smiles": None, "error": f"http {r.status_code}"}
        data = r.json()
        smiles = (data.get("molecule_structures") or {}).get("canonical_smiles")
        if not smiles:
            return {"id": chembl_id, "smiles": None, "error": "no canonical_smiles"}
        return {"id": chembl_id, "smiles": smiles, "error": None}
    except requests.RequestException as e:
        return {"id": chembl_id, "smiles": None, "error": str(e)}


def fetch_uniprot_fasta(uniprot_id: str, timeout: float = 20.0) -> dict:
    """Return: {"id": uniprot_id, "fasta": <str|None>, "error": <str|None>}"""
    url = f"{UNIPROT_BASE}/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=timeout, headers={"Accept": "text/x-fasta"})
        if r.status_code != 200:
            return {"id": uniprot_id, "fasta": None, "error": f"http {r.status_code}"}
        text = r.text.strip()
        if not text.startswith(">"):
            return {"id": uniprot_id, "fasta": None, "error": "invalid fasta payload"}
        return {"id": uniprot_id, "fasta": text, "error": None}
    except requests.RequestException as e:
        return {"id": uniprot_id, "fasta": None, "error": str(e)}


def fetch_pdb_file(pdb_id: str, out_dir: str = "data/raw/pdb", fmt: str = "pdb", timeout: float = 30.0) -> dict:
    """
    Download PDB/mmCIF from RCSB.
    Return: {"id": pdb_id, "path": <str|None>, "error": <str|None>}
    """
    pdb_code = pdb_id.upper().strip()
    if fmt not in {"pdb", "cif"}:
        return {"id": pdb_id, "path": None, "error": "fmt must be 'pdb' or 'cif'"}
    url = f"{RCSB_FILE_BASE}/{pdb_code}.{fmt}"
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / f"{pdb_code}.{fmt}"
    try:
        r = requests.get(url, timeout=timeout, stream=True)
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
    p.add_argument("--pdb-fmt", choices=["pdb", "cif"], default="pdb", help="PDB download format")
    p.add_argument("--pdb-outdir", default="data/raw/pdb", help="Directory to save PDB files")
    p.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    p.set_defaults(func=run)


def run(args):
    result = {
        "chembl": {},
        "uniprot": {},
        "pdb": {},
        "meta": {"ok": True, "errors": []},
    }

    # ChEMBL
    for cid in args.chembl:
        rec = fetch_chembl_smiles(cid)
        result["chembl"][cid] = rec["smiles"]
        if rec["error"]:
            result["meta"]["errors"].append({"chembl": cid, "msg": rec["error"]})

    # UniProt
    for uid in args.uniprot:
        rec = fetch_uniprot_fasta(uid)
        result["uniprot"][uid] = rec["fasta"]
        if rec["error"]:
            result["meta"]["errors"].append({"uniprot": uid, "msg": rec["error"]})

    # PDB
    for pid in args.pdb:
        rec = fetch_pdb_file(pid, out_dir=args.pdb_outdir, fmt=args.pdb_fmt)
        result["pdb"][pid] = rec["path"]
        if rec["error"]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
