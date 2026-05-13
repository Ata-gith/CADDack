# tests/test_fetch_structures.py
import types, json
from pathlib import Path
from scripts import fetch_structures as fs

class DummyResp:
    def __init__(self, code, text="", json_data=None):
        self.status_code = code
        self.text = text
        self._json = json_data or {}
    def json(self): return self._json
    def iter_content(self, chunk_size=8192):
        yield b"ATOM  "  # minimal payload

def test_fetch_pdb_file_writes_under_repo_root(monkeypatch, tmp_path):
    def fake_get(url, *, timeout, stream=False, headers=None):
        return DummyResp(200, text="ATOM")
    monkeypatch.setattr(fs, "_get", fake_get)

    fs.REPO_ROOT = tmp_path
    out = fs.fetch_pdb_file("1CRN")
    assert out["error"] is None
    assert (tmp_path / "data" / "raw" / "pdb" / "1CRN.pdb").exists()

def test_full_run_emits_files_and_index(monkeypatch, tmp_path, capsys):
    # mock all endpoints, distinguishing molecule vs activity ChEMBL routes
    def fake_get(url, *, timeout, stream=False, headers=None):
        if "activity" in url:
            return DummyResp(200, json_data={"activities": [{"pchembl_value": "7.0"}]})
        if "molecule/" in url:
            return DummyResp(200, json_data={"molecule_structures": {"canonical_smiles": "CCO"}})
        if url.endswith(".fasta"):
            return DummyResp(200, text=">hdr\nSEQUENCE")
        if url.endswith(".pdb"):
            return DummyResp(200, text="ATOM")
        return DummyResp(404)
    monkeypatch.setattr(fs, "_get", fake_get)

    fs.REPO_ROOT = tmp_path

    args = types.SimpleNamespace(
        chembl=["CHEMBL25"],
        chembl_target=None,     # no bulk target harvest
        chembl_ids_file=None,
        uniprot=["P00734"],
        pdb=["1CRN"],
        pdb_fmt="pdb",
        pdb_outdir=None,        # default: <REPO_ROOT>/data/raw/pdb
        emit_mols_csv=str(tmp_path / "data" / "raw" / "mols.csv"),
        out=str(tmp_path / "data" / "processed" / "fetch_index.json"),
    )

    fs.run(args)

    # Check files exist
    smi = tmp_path / "data" / "raw" / "chembl" / "CHEMBL25.smi"
    fasta = tmp_path / "data" / "raw" / "uniprot" / "P00734.fasta"
    pdbf = tmp_path / "data" / "raw" / "pdb" / "1CRN.pdb"
    mols = tmp_path / "data" / "raw" / "mols.csv"
    index = tmp_path / "data" / "processed" / "fetch_index.json"

    assert smi.exists() and fasta.exists() and pdbf.exists()
    assert mols.exists() and index.exists()

    # Validate index JSON
    data = json.loads(index.read_text(encoding="utf-8"))
    assert data["meta"]["ok"] is True
    assert "chembl" in data and "uniprot" in data and "pdb" in data
    assert data["chembl"]["CHEMBL25"]["smiles_path"].endswith("CHEMBL25.smi")
    assert data["uniprot"]["P00734"]["fasta_path"].endswith("P00734.fasta")
    assert data["pdb"]["1CRN"]["path"].endswith("1CRN.pdb")

    # Validate mols.csv content
    txt = mols.read_text(encoding="utf-8").strip().splitlines()
    assert txt[0].strip() == "SMILES,chembl_id,pIC50"
    assert "CCO,CHEMBL25" in txt[1]
