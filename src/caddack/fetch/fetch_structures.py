#!/usr/bin/env python3
# scripts/fetch_structures.py
import argparse
import csv
import json
import math
import statistics as stats
import sys
from pathlib import Path

import requests

# --- Endpoints ---
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_FILE_BASE = "https://files.rcsb.org/download"  # {PDB}.pdb or {PDB}.cif

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "CADDack/0.1 (+https://github.com/Ata-gith/CADDack)"}

# -------------------- IO helpers --------------------
def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def save_chembl(chembl_id: str, smiles: str) -> str:
    chembl_id = chembl_id.upper().strip()
    p = REPO_ROOT / "data" / "raw" / "chembl" / f"{chembl_id}.smi"
    _write_text(p, smiles + "\n")
    return str(p)

def save_uniprot(uniprot_id: str, fasta: str) -> str:
    uniprot_id = uniprot_id.upper().strip()
    p = REPO_ROOT / "data" / "raw" / "uniprot" / f"{uniprot_id}.fasta"
    _write_text(p, fasta + ("\n" if not fasta.endswith("\n") else ""))
    return str(p)

def _read_ids_file(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out

# -------------------- HTTP --------------------
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

# -------------------- ChEMBL helpers --------------------
def _chembl_paged(url_base: str, page_size: int = 1000, max_pages: int = 20, timeout: float = 20.0):
    """
    Yield JSON 'activities' pages from a ChEMBL collection endpoint using offset paging.
    Stops on empty page or max_pages.
    """
    for i in range(max_pages):
        url = f"{url_base}&limit={page_size}&offset={i*page_size}"
        r = _get(url, timeout=timeout)
        if r.status_code != 200:
            break
        data = r.json() or {}
        items = data.get("activities") or []
        if not items:
            break
        yield items

def fetch_target_pic50s(target_chembl_id: str, needed: int = 50, timeout: float = 20.0) -> dict[str, float]:
    """
    Return dict {molecule_chembl_id: median_pIC50} for a ChEMBL target,
    collecting until 'needed' unique molecules (or data exhausted).
    """
    target_chembl_id = target_chembl_id.upper().strip()
    base = (f"{CHEMBL_BASE}/activity.json?"
            f"target_chembl_id={target_chembl_id}&standard_type=IC50")
    per_mol: dict[str, list[float]] = {}
    for page in _chembl_paged(base, page_size=1000, max_pages=50, timeout=timeout):
        for a in page:
            mid = (a.get("molecule_chembl_id") or "").upper().strip()
            if not mid:
                continue
            # prefer pChEMBL
            val = None
            pc = a.get("pchembl_value")
            if pc is not None:
                try:
                    val = float(pc)
                except Exception:
                    val = None
            if val is None:
                rel = (a.get("relation") or a.get("standard_relation") or "=").strip()
                if rel in ("=", "~", "<", "<="):
                    units = a.get("standard_units")
                    v = a.get("standard_value")
                    if units and v is not None:
                        try:
                            vv = float(v)
                            val = _ic50_to_pIC50(vv, units)
                        except Exception:
                            val = None
            if val is None or not math.isfinite(val):
                continue
            per_mol.setdefault(mid, []).append(val)
        if len(per_mol) >= needed:
            break
    out: dict[str, float] = {}
    for mid, vals in per_mol.items():
        try:
            out[mid] = float(stats.median(vals))
        except Exception:
            pass
    return out

def fetch_chembl_smiles(chembl_id: str, timeout: float = 20.0) -> dict:
    """
    Return: {"id": chembl_id, "smiles": <str|None>, "error": <str|None>}
    """
    chembl_id = chembl_id.upper().strip()
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
    Supported units:
      nM: pIC50 = 9 - log10(IC50_nM)
      uM: pIC50 = 6 - log10(IC50_uM)
       M: pIC50 = -log10(IC50_M)
    """
    if value <= 0:
        return None
    u = (units or "").strip().upper()
    if u == "NM":
        return 9.0 - math.log10(value)
    if u in ("UM", "ΜM"):
        return 6.0 - math.log10(value)
    if u == "M":
        return -math.log10(value)
    return None

def fetch_chembl_pIC50(chembl_id: str, timeout: float = 20.0) -> dict:
    """
    Fetch IC50 activities and aggregate to median pIC50.
    Return: {"id": chembl_id, "pIC50": <float|None>, "error": <str|None>}
    """
    chembl_id = chembl_id.upper().strip()
    url = (
        f"{CHEMBL_BASE}/activity.json?"
        f"molecule_chembl_id={chembl_id}&standard_type=IC50&limit=1000"
    )
    try:
        r = _get(url, timeout=timeout)
        if r.status_code != 200:
            return {"id": chembl_id, "pIC50": None, "error": f"http {r.status_code}"}
        data = r.json() or {}
        acts = data.get("activities") or []

        vals: list[float] = []
        for a in acts:
            pc = a.get("pchembl_value")
            if pc is not None:
                try:
                    v = float(pc)
                    if math.isfinite(v):
                        vals.append(v)
                        continue
                except (TypeError, ValueError):
                    pass
            stype = (a.get("standard_type") or "").upper()
            if stype != "IC50":
                continue
            rel = (a.get("standard_relation") or a.get("relation") or "=").strip()
            if rel not in ("=", "~", "<", "<="):
                continue
            units = a.get("standard_units")
            val = a.get("standard_value")
            try:
                fv = float(val) if val is not None else None
            except (TypeError, ValueError):
                fv = None
            if fv is None:
                continue
            p = _ic50_to_pIC50(fv, units or "")
            if p is not None and math.isfinite(p):
                vals.append(p)

        if not vals:
            return {"id": chembl_id, "pIC50": None, "error": "no_usable_ic50"}
        return {"id": chembl_id, "pIC50": float(stats.median(vals)), "error": None}
    except requests.RequestException as e:
        return {"id": chembl_id, "pIC50": None, "error": str(e)}

# -------------------- UniProt --------------------
def fetch_uniprot_fasta(uniprot_id: str, timeout: float = 20.0) -> dict:
    """
    Return: {"id": uniprot_id, "fasta": <str|None>, "error": <str|None>}
    """
    uniprot_id = uniprot_id.upper().strip()
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

# -------------------- RCSB PDB --------------------
def fetch_pdb_file(
    pdb_id: str,
    out_dir: str | None = None,
    fmt: str = "pdb",
    timeout: float = 30.0,
) -> dict:
    """
    Download PDB/mmCIF into <repo>/data/raw/pdb by default.
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

# -------------------- CLI --------------------
def add_cli(subparsers):
    p = subparsers.add_parser(
        "fetch",
        help="Fetch structures from public databases (ChEMBL, UniProt, PDB)",
    )
    # IDs
    p.add_argument("--chembl", nargs="*", default=[], help="ChEMBL molecule IDs")
    p.add_argument("--chembl-ids-file", default=None, help="Text file with ChEMBL IDs (one per line)")
    p.add_argument("--chembl-target", default=None, help="ChEMBL target ID (e.g., CHEMBL203) to auto-collect ligands")
    p.add_argument("--min-n", type=int, default=50, help="Minimum number of ligands to collect when using --chembl-target")
    p.add_argument("--min-pchembl", type=float, default=None, help="Optional pChEMBL/pIC50 threshold for target harvesting")

    p.add_argument("--uniprot", nargs="*", default=[], help="UniProt accessions")
    p.add_argument("--pdb", nargs="*", default=[], help="PDB IDs")
    p.add_argument("--pdb-fmt", choices=["pdb", "cif"], default="pdb", help="PDB download format")
    p.add_argument("--pdb-outdir", default=str(REPO_ROOT / "data" / "raw" / "pdb"), help="Directory to save PDB files")

    # Outputs
    p.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    p.add_argument("--emit-mols-csv", default=None, help="Write CSV of ligands (SMILES,chembl_id,pIC50)")
    p.set_defaults(func=run)

def run(args):
    # Auto-collect ChEMBL ligands for a target to reach at least N rows
    auto_cids: list[tuple[str, float]] = []
    if args.chembl_target:
        pic50_map = fetch_target_pic50s(args.chembl_target, needed=args.min_n)
        # Order by strongest (highest pIC50) then truncate to min_n
        auto_cids = sorted(pic50_map.items(), key=lambda kv: kv[1], reverse=True)
        auto_cids = auto_cids[: args.min_n]
        # merge with explicit --chembl list (explicit first)
        explicit = [c.upper().strip() for c in (args.chembl or [])]
        for mid, _ in auto_cids:
            if mid not in explicit:
                explicit.append(mid)
        args.chembl = explicit

    # Build ChEMBL ID list (file → explicit), dedup, keep order
    chembl_ids: list[str] = []
    chembl_ids.extend(_read_ids_file(getattr(args, "chembl_ids_file", None)))
    chembl_ids.extend(args.chembl or [])
    seen = set()
    norm_ids: list[str] = []
    for x in chembl_ids:
        xx = x.upper().strip()
        if xx and xx not in seen:
            seen.add(xx)
            norm_ids.append(xx)
    chembl_ids = norm_ids

    result = {"chembl": {}, "uniprot": {}, "pdb": {}, "meta": {"ok": True, "errors": []}}
    chembl_rows: list[dict] = []

    # ChEMBL
    pic50_from_target = {mid: val for mid, val in (auto_cids or [])}
    for cid in chembl_ids:
        cid_norm = cid.upper().strip()
        rec_smiles = fetch_chembl_smiles(cid_norm)
        if rec_smiles["smiles"]:
            path = save_chembl(cid_norm, rec_smiles["smiles"])
            result["chembl"][cid_norm] = {"smiles_path": path}
            # pIC50: use target-derived if present; else fetch directly
            if cid_norm in pic50_from_target:
                pic50 = pic50_from_target[cid_norm]
            else:
                rec_pic50 = fetch_chembl_pIC50(cid_norm)
                pic50 = rec_pic50["pIC50"]
                if rec_pic50["error"]:
                    result["meta"]["errors"].append({"chembl_pIC50": cid_norm, "msg": rec_pic50["error"]})
            if pic50 is not None:
                result["chembl"][cid_norm]["pIC50"] = pic50
            chembl_rows.append({"SMILES": rec_smiles["smiles"], "chembl_id": cid_norm, "pIC50": pic50})
        else:
            result["chembl"][cid_norm] = None
            result["meta"]["errors"].append({"chembl": cid_norm, "msg": rec_smiles["error"]})

    # UniProt
    for uid in args.uniprot:
        uid = uid.upper().strip()
        rec = fetch_uniprot_fasta(uid)
        if rec["fasta"]:
            path = save_uniprot(uid, rec["fasta"])
            result["uniprot"][uid] = {"fasta_path": path}
        else:
            result["uniprot"][uid] = None
            result["meta"]["errors"].append({"uniprot": uid, "msg": rec["error"]})

    # PDB
    for pid in args.pdb:
        pid = pid.upper().strip()
        rec = fetch_pdb_file(pid, out_dir=args.pdb_outdir, fmt=args.pdb_fmt)
        if rec["path"]:
            result["pdb"][pid] = {"path": rec["path"]}
        else:
            result["pdb"][pid] = None
            result["meta"]["errors"].append({"pdb": pid, "msg": rec["error"]})

    # finalize ok flag
    if result["meta"]["errors"]:
        result["meta"]["ok"] = False

    # write JSON index
    payload = json.dumps(result, indent=2)
    if args.out == "-" or args.out.strip() == "":
        sys.stdout.write(payload + "\n")
    else:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(payload, encoding="utf-8")

    # optional ligands CSV
    if args.emit_mols_csv:
        out_csv = Path(args.emit_mols_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["SMILES", "chembl_id", "pIC50"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in chembl_rows:
                if r.get("pIC50") is None:
                    r["pIC50"] = ""
                w.writerow(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="caddack", description="CADDack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    args.func(args)
