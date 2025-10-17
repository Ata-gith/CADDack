import types
from scripts import fetch_structures as fs

class DummyResp:
    def __init__(self, code, text=None, json_data=None):
        self.status_code = code
        self.text = text or ""
        self._json = json_data or {}
    def json(self): return self._json
    def iter_content(self, chunk_size=8192): yield b"data"

def test_fetch_chembl_smiles(monkeypatch):
    def fake_get(url, timeout): return DummyResp(200, json_data={"molecule_structures": {"canonical_smiles": "CCO"}})
    monkeypatch.setattr(fs, "requests", types.SimpleNamespace(get=fake_get))
    out = fs.fetch_chembl_smiles("CHEMBL25")
    assert out["smiles"] == "CCO" and out["error"] is None

def test_fetch_uniprot(monkeypatch):
    def fake_get(url, timeout, headers=None): return DummyResp(200, text=">Header\nSEQUENCE")
    monkeypatch.setattr(fs, "requests", types.SimpleNamespace(get=fake_get))
    out = fs.fetch_uniprot_fasta("P00734")
    assert out["fasta"].startswith(">") and out["error"] is None

def test_fetch_pdb(monkeypatch, tmp_path):
    def fake_get(url, timeout, stream=True):
        return DummyResp(200, text="MODEL", json_data=None)
    monkeypatch.setattr(fs, "requests", types.SimpleNamespace(get=fake_get))
    out = fs.fetch_pdb_file("1CRN", out_dir=tmp_path)
    assert "1CRN" in out["path"] and out["error"] is None
