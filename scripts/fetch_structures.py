import argparse
import json
import sys
from pathlib import Path
import requests

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_chembl_smiles(chembl_id: str, timeout: float = 20.0) -> dict:
    """
    Return dict: {"id": chembl_id, "smiles": <str or None>, "error": <str or None>}
    """
    url = f"{CHEMBL_BASE}/molecule/{chembl_id}.json"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {"id": chembl_id, "smiles": None, "error": f"http {r.status_code}"}
        data = r.json()
        smiles = (
            data.get("molecule_structures", {}) or {}
        ).get("canonical_smiles", None)
        if not smiles:
            return {"id": chembl_id, "smiles": None, "error": "no canonical_smiles"}
        return {"id": chembl_id, "smiles": smiles, "error": None}
    except requests.RequestException as e:
        return {"id": chembl_id, "smiles": None, "error": str(e)}

def add_cli(subparsers):
    p = subparsers.add_parser(
        "fetch",
        help="Fetch structures from public databases (ChEMBL, UniProt, PDB)",
    )
    p.add_argument("--chembl", nargs="*", default=[], help="ChEMBL IDs")
    p.add_argument("--uniprot", nargs="*", default=[], help="UniProt accessions")
    p.add_argument("--pdb", nargs="*", default=[], help="PDB IDs")
    p.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    p.set_defaults(func=run)

def run(args):
    result = {
        "chembl": {},
        "uniprot": {},
        "pdb": {},
        "meta": {"ok": True, "errors": []},
    }

    # ChEMBL: fetch canonical SMILES
    for cid in args.chembl:
        rec = fetch_chembl_smiles(cid)
        result["chembl"][cid] = rec["smiles"]
        if rec["error"]:
            result["meta"].setdefault("errors", []).append({"chembl": cid, "msg": rec["error"]})

    # Placeholders for next steps
    for uid in args.uniprot:
        result["uniprot"][uid] = None
    for pid in args.pdb:
        result["pdb"][pid] = None

    if result["meta"].get("errors"):
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