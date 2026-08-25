# Changelog

History of changes relative to the original `gpt` branch.

## Bug fixes

**Correctness audit of `main` (GNN / geometry / QSAR / fetch)**
- `bayes.elbo_loss`: KL is scaled by the dataset size `n_train`, not the batch
  count — dividing by batches over-weighted the KL by ~`batch_size`, collapsing
  the posterior toward the prior. Also restored the parameter name the tests use.
- `geometry.load_ligand` / `load_complex`: `.mol2` now uses `Chem.MolFromMol2File`
  (Tripos MOL2) instead of the MDL-molfile reader, which silently returned `None`
  and dropped every `.mol2` complex.
- `geometry._element_from_name`: column-aware element inference for blank element
  columns — `" CA "`→carbon (was calcium), `" ND1"`→N, `" OD1"`→O (were carbon).
- `gnn.train._prepare_frame`: `task="classification"` with a continuous target and
  no `positive_threshold` now raises a clear error instead of training on floats.
- `train_from_csv` / `qsar-train`: `roc_auc` is omitted (not crashed) on a
  single-class test split.
- `datasets.smiles_to_graph_arrays`: 0-atom mols (e.g. the empty string) are
  rejected so `skip_invalid` drops them instead of emitting a malformed tensor.
- `models.predict_with_uncertainty`: no longer returns `NaN` at `n_samples=1`
  (biased variance) and restores the caller's train/eval mode.
- `fetch`: `--min-pchembl` is now applied (was ignored); censored IC50s
  (`<`, `<=`) are excluded from the pIC50 median; HTTP retries use backoff.
- `qsar-descriptors`: featurizes without silently dropping rows on the `pIC50`
  column. `qsar-train --max-features` accepts numeric values.
- Fusion tests no longer require `torch_scatter` (the tower has pure-PyTorch
  fallbacks), so they actually run in scatter-less environments.

**`tests/test_fetch_structures.py`**
- The PDB assertion accessed `data["pdb"]["1CRN"]` as a bare string; it is a dict
  `{"path": ...}`. Fixed to `data["pdb"]["1CRN"]["path"].endswith(...)`.
- The expected CSV header was `"SMILES,chembl_id"` but the writer includes a third column;
  fixed to `"SMILES,chembl_id,pIC50"`.
- The mock `fake_get` routed all ChEMBL URLs on `"chembl" in url`, so the pIC50 fetch
  received the wrong response payload. Fixed to route on specific URL patterns
  (`"activity" in url` vs `"molecule/" in url`).
- `args` SimpleNamespace was missing `chembl_target` and `chembl_ids_file`; added both as
  `None`.

**`scripts/run_qsar_descriptors.py`**
- Simplified the `--drop-errors` filter from a convoluted multi-condition expression to
  `feats[feats["__error"] != "invalid_smiles"]`.

## Improvements

**`src/caddack/qsar/descriptors.py`** — lazy RDKit imports
- Removed top-level `from rdkit import Chem` (caused `ImportError` on import without RDKit).
- Added `_require_rdkit()` helper that imports at call-time and raises with a clear install
  hint.
- Moved the descriptor list into `_basic_desc()` so RDKit callables resolve lazily.
- Added `_get_rfg()` for the modern `rdFingerprintGenerator` API with a `None` fallback to
  the legacy `GetMorganFingerprintAsBitVect` — supports both old and new RDKit versions.

**`src/caddack/qsar/split.py`** — lazy imports + scaffold-split correctness
- Same lazy `_require_rdkit()` pattern.
- `murcko_scaffold` previously returned `""` for invalid SMILES, grouping all bad molecules
  under the same scaffold key and silently distorting splits. Fixed to return `None`.
- `scaffold_split` now explicitly routes `None`-scaffold rows to the **train set**.
- Fixed default `seed` from `40` → `42`.

**`tests/test_imports.py`** — conditional tests
- All tests are now gated on `importlib.util.find_spec` checks so the suite passes whether
  or not RDKit / torch are installed.
- Added positive tests for `parse_smiles`, `smiles_to_features`, `strip_salts`,
  `murcko_scaffold`, `scaffold_split`, and the corresponding absence-tests that verify
  `ImportError` is raised at call-time.

## New: Two-tower Bayesian fusion model

**`src/caddack/gnn/bayes.py`** — `BayesianLinear`, `BayesianMLP`, `elbo_loss`

Variational BNN components in pure PyTorch. No new external dependencies.

**`src/caddack/gnn/geometry.py`** — PDB parser + complex loader

Fixed-width PDB ATOM/HETATM parser that needs no Biopython. Parses protein pockets,
loads ligand .sdf/.mol2 files, and assembles `ComplexExample` / `GeometryRecord` containers
for the training pipeline.

**`src/caddack/gnn/models.py`** — `GaussianRBF`, `GeometryTower`, `FusionAffinityNet`

SchNet-style geometry tower + two-tower fusion model added alongside the existing
`MolecularGCN` and `MolecularGINE` classes. All use the same lazy-torch factory pattern.

**`src/caddack/gnn/train.py`** — `train_fusion_from_complexes`

End-to-end training loop for `FusionAffinityNet`: two-graph batching with a custom
`collate_fn`, ELBO training with KL warmup, MC sampling at evaluation, and saving
`model.pt` + `metrics.json` + `config.json`.

**`src/caddack/gnn/__init__.py`** — updated exports

Exports `FusionAffinityNet`, `ComplexExample`, `GeometryRecord`, `load_complex_dataset`,
`train_fusion_from_complexes`.

**`scripts/train_fusion.py`** — `train-fusion` CLI subcommand

**`tests/test_fusion.py`** — tests for the fusion pipeline

No-dep tests (PDB parser, import guards) run without any optional packages. BNN unit tests,
forward/backward/ELBO, and the training smoke test are gated on `torch + pyg + rdkit +
torch-scatter`.

**`pyproject.toml`** — populated dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `requests`, `joblib`.
`[gnn]` extra: `torch`, `torch-geometric`, `torch-scatter`, `rdkit`.
`[dev]` extra: `pytest`.

---

