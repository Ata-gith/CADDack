# CADDack

Modular, scriptable toolkit for end-to-end computer-aided drug design (CADD). Covers data
fetching, QSAR featurisation, 2D message-passing GNNs, and a two-tower Bayesian
affinity model that fuses 2D ligand graphs with 3D protein–ligand geometry.

---

## Repository layout

```
CADDack/
├── configs/                  YAML configs for CLI runs
├── data/                     raw/ and processed/ data (gitignored except fixtures)
├── docs/                     design notes
├── scripts/                  CLI entry-points that call into src/caddack/
├── src/caddack/              installable package
│   ├── qsar/                 descriptor featurisation + scaffold splitting
│   └── gnn/                  graph neural networks + Bayesian fusion model
├── tests/                    pytest suites
├── pyproject.toml
└── environment.yml
```

---

## Install

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate caddack
pip install -e .
```

### Pip (core only, no GNN)

```bash
pip install -e .
```

### Pip with GNN + RDKit support

```bash
pip install -e ".[gnn]"
```

Optional dependencies are intentionally lazy — every module can be **imported** without them.
Errors are only raised at call-time with a clear install hint.

---

## Modules

### `caddack.qsar` — QSAR descriptors and dataset splitting

#### `descriptors.py`

Physicochemical + fingerprint featurisation of SMILES strings.

| Function | Description |
|---|---|
| `parse_smiles(smiles)` | Parse SMILES → RDKit `Mol` or `None` if invalid. |
| `canonicalize_smiles(smiles)` | Return canonical SMILES or `None`. |
| `strip_salts(smiles)` | Keep the largest fragment (removes counter-ions). |
| `smiles_to_mol_clean(smiles)` | Strip salts → canonicalise → parse, all in one step. |
| `mol_to_basic_features(mol)` | `dict` of 6 physicochemical descriptors: `MolWt`, `LogP`, `TPSA`, `NumHBD`, `NumHBA`, `NumRotBonds`. |
| `mol_to_ecfp_bits(mol, radius, n_bits)` | `dict` of ECFP bit columns (`ECFP2_0` … `ECFP2_2047`). Supports both modern `rdFingerprintGenerator` API and the legacy `GetMorganFingerprintAsBitVect` fallback. |
| `smiles_to_features(smiles, radius, n_bits)` | Full feature dict: physicochemical + ECFP + `SMILES_canonical`. Returns `None` for invalid SMILES. |
| `featurize_dataframe(df, smiles_col, radius, n_bits, target_col, drop_na_target)` | Featurise an entire DataFrame. Rows with invalid SMILES get `__error = "invalid_smiles"` instead of raising. Coerces target to numeric and optionally drops NaN targets. |

All RDKit calls are wrapped in a lazy `_require_rdkit()` helper — the module imports cleanly
without RDKit installed.

#### `split.py`

Scaffold-aware train/test splitting.

| Function | Description |
|---|---|
| `murcko_scaffold(smiles)` | Return canonical Murcko scaffold SMILES, or `None` for invalid SMILES (never returns an empty string — that would silently group all acyclics under one key). |
| `scaffold_split(df, smiles_col, test_size, seed)` | Group molecules by Murcko scaffold, shuffle scaffold groups, fill test set up to `test_size` fraction. Rows with invalid SMILES always go to **train** to prevent bad data from leaking into evaluation. Returns `(train_idx, test_idx)` as numpy integer arrays. |

---

### `caddack.gnn` — Graph Neural Networks

#### `datasets.py`

Framework-agnostic molecular graph builder.

| Symbol | Description |
|---|---|
| `GraphArrays` | Dataclass holding `node_features`, `edge_index`, `edge_features` as plain Python lists — no torch dependency. |
| `smiles_to_graph_arrays(smiles)` | Build a `GraphArrays` from SMILES. Atom features (7-dim): atomic number, degree, formal charge, aromaticity, hybridisation, H count, ring membership. Bond features (7-dim): bond type (×4 one-hot), conjugation, ring membership, stereo. Edges are bidirectional. |
| `to_pyg_data(graph, y)` | Convert `GraphArrays` → `torch_geometric.data.Data` with optional label `y`. |
| `build_pyg_dataset(smiles_list, targets)` | Convenience wrapper: build a list of `Data` objects from parallel SMILES + target lists. |

#### `models.py`

All models use the **factory pattern**: a class with a `build(...)` static method that
instantiates the real `nn.Module` only when torch is available. This keeps the module
importable without torch.

##### `MolecularGCN`

Simple graph convolutional network (GCNConv + global mean pool). Suitable for fast
classification or regression baselines.

```
build(in_channels, hidden_channels=128, num_layers=3, dropout=0.1, task="classification")
```

##### `MolecularGINE`

Graph Isomorphism Network with Edge features (GINEConv + global add pool). Handles bond
features that GCN ignores. Recommended for ligand-only property prediction.

```
build(in_channels, edge_dim, hidden_channels=128, num_layers=3, dropout=0.1, task="classification")
```

##### `GaussianRBF`

Gaussian radial basis function expansion. Converts a scalar interatomic distance into a
`num_rbf`-dimensional feature vector by placing Gaussian kernels evenly spaced along
`[0, cutoff]`. Used inside the geometry tower.

```
build(num_rbf=50, cutoff=6.0)
```

##### `GeometryTower`

SE(3)-invariant SchNet-style continuous-filter convolutional tower over interatomic distances.
Invariance is guaranteed because it operates **only on pairwise distances**, never on raw
(x, y, z) coordinates.

```
build(num_elements=100, hidden_channels=128, num_rbf=50, num_interactions=3, cutoff=6.0)
```

Architecture per forward pass:
1. Atom type → embedding (`nn.Embedding(z)`)
2. Build `radius_graph` (computed **inside** `forward` from `pos` + `batch` — no pre-computed edge_index stored on the dataset)
3. Expand distances with `GaussianRBF`
4. Apply `num_interactions` `InteractionBlock`s, each doing: filter network(rbf) × cosine-cutoff(dist) → weighted messages → scatter-add → atom MLP → residual update
5. `global_add_pool` over the batch to get a single vector per graph

The `radius_graph`-inside-forward design solves the **two-graph batching problem**: the
geometry graph and the ligand 2D graph have different node counts per example, so they
can't share one PyG `Data` object. Computing the geometry edge_index on-the-fly from
the batch tensor means no geometry edge_index ever needs PyG collation.

##### `FusionAffinityNet`

Two-tower Bayesian fusion model for protein–ligand binding affinity with uncertainty.

```
build(
    ligand_in_channels=7,   ligand_edge_dim=7,
    hidden_channels=128,
    num_gine_layers=3,      num_geo_interactions=3,
    num_rbf=50,             cutoff=6.0,
    num_elements=100,
    bayesian_hidden=[256, 128],
    prior_sigma=1.0,        dropout=0.1,
)
```

**Architecture:**

```
ligand SMILES ──► Tower 1 (GINE, 2D atom/bond feats) ──► h_lig [B, 128]
                                                                  │ concat
protein PDB + ligand ──► pocket atoms (≤ cutoff Å) ──► Tower 2 (SchNet distances) ──► h_geo [B, 128]
                                                                  │
                              concat([h_lig, h_geo]) ──► Bayesian MLP head ──► (μ, log σ²)
```

- Tower 1 is the existing GINE stack; Tower 2 is the new geometry tower.
- Only the **fusion head** is Bayesian (last-layer Bayes-by-Backprop) — towers are
  deterministic. This keeps the KL tractable and training stable.
- `forward(lig_data, geo_data)` returns `(mu, log_var)`.
- `predict_with_uncertainty(lig_data, geo_data, n_samples=30)` runs `n_samples` Monte Carlo
  forward passes and returns `(pred_mean, epistemic_std, aleatoric_std)`:
  - **Epistemic** uncertainty = variance of the sampled means (what the model doesn't know)
  - **Aleatoric** uncertainty = mean of the sampled variances (irreducible noise in the data)

#### `bayes.py`

Pure PyTorch variational Bayes components. No new external dependencies.

##### `BayesianLinear`

A drop-in replacement for `nn.Linear` with learned weight distributions.

- Stores `weight_mu` and `weight_rho` parameters per weight and bias.
- `σ = softplus(rho)` — strictly positive, smooth, initialised near zero (rho = −5).
- Forward pass: `w = μ + σ · ε` where `ε ~ N(0, I)` (reparameterization trick).
- `kl()`: closed-form KL divergence to the isotropic Gaussian prior `N(0, prior_σ²)`:
  ```
  KL = Σ [ ln(prior_σ/σ) + (σ² + μ²) / (2·prior_σ²) − 0.5 ]
  ```

##### `BayesianMLP`

Stack of `BayesianLinear` layers with ReLU activations, followed by two deterministic
linear output heads producing `(mu, log_var)` — the predicted mean and log-variance of the
affinity distribution.

##### `elbo_loss(mu, log_var, y, kl, n_train, kl_weight, aleatoric)`

ELBO training objective:

```
ELBO = E_q[NLL] + β · KL / N_train
```

- **Heteroscedastic NLL** (when `aleatoric=True`):
  `NLL = 0.5 · mean[ exp(−s)·(y − μ)² + s ]`  where `s = log_var`, clamped to [−10, 10].
  This lets the model predict its own output noise per example.
- Falls back to **MSE** when `aleatoric=False`.
- `β = kl_weight · min(1, epoch / kl_warmup)` (KL annealing) — β ramps linearly from 0 to
  `kl_weight` over `kl_warmup` epochs, preventing the KL-collapse failure mode where
  `σ → 0` and the head degenerates to the prior mean.

#### `geometry.py`

Dependency-light 3D complex parser and dataset loader. The PDB parser needs no Biopython.

| Symbol | Description |
|---|---|
| `PDBAtom` | Dataclass for one parsed atom: record type, name, residue, chain, coordinates, element, atomic number. |
| `parse_pdb_atoms(pdb_path)` | Fixed-width PDB ATOM/HETATM parser. Takes only the first MODEL; skips alternate locations (altLoc B/C/…). Infers element from atom name when the element column is blank. Returns `[]` for missing or unreadable files — never raises. |
| `load_ligand(ligand_path)` | Load ligand from `.sdf` or `.mol2` via RDKit, returning a list of `PDBAtom` with 3D conformer coordinates. Returns `None` if RDKit is unavailable or parsing fails. |
| `extract_pocket(protein_atoms, ligand_atoms, cutoff)` | Return the subset of `ATOM` records within `cutoff` Å of any ligand atom — the binding pocket. |
| `GeometryRecord` | Dataclass: `atomic_nums [N]`, `positions [N]` (x,y,z tuples), `n_ligand`, `n_pocket`. Ligand atoms come first, then pocket atoms — this order is documented to avoid off-by-atom bugs when slicing by source. |
| `ComplexExample` | Dataclass for one protein–ligand complex: `pdb_id`, `affinity`, `ligand_smiles` (for Tower 1), `geo: GeometryRecord` (for Tower 2). |
| `load_complex(pdb_path, ligand_path, affinity, pdb_id, cutoff)` | Load one complex from a PDB + ligand file pair. Returns `None` if parsing yields fewer than 1 ligand or pocket atom. |
| `load_complex_dataset(index_csv, root, ...)` | Load a PDBbind-style dataset. Expects `root/<pdb_id>/<pdb_id>_protein.pdb` and `root/<pdb_id>/<pdb_id>_ligand.sdf`. Rows that fail are silently skipped. |

#### `train.py`

Training loops.

| Function | Description |
|---|---|
| `train_from_csv(csv_path, smiles_col, target_col, ...)` | Train `MolecularGCN` or `MolecularGINE` on a SMILES CSV. Saves `model.pt` + `metrics.json`. Returns metric dict. |
| `train_fusion_from_complexes(complexes, outdir, ...)` | Train `FusionAffinityNet` on a list of `ComplexExample` objects. Uses scaffold split on ligand SMILES when available, else random. Pairs each complex into two parallel batches (ligand 2D + geometry), trains with ELBO, evaluates with MC sampling. Saves `model.pt`, `metrics.json`, and `config.json` (all arch hyperparams needed to reconstruct the model). |

---

## CLI subcommands

All subcommands are registered through `src/caddack/cli.py` and available as:

```bash
caddack <subcommand> [options]
```

### `fetch` — Data fetching

```bash
caddack fetch \
  --chembl CHEMBL25 \          # one or more ChEMBL compound IDs
  --chembl-target CHEMBL203 \  # bulk-harvest all actives for a target
  --uniprot P00734 \           # UniProt FASTA
  --pdb 1CRN \                 # PDB structure (.pdb or .cif)
  --emit-mols-csv data/raw/mols.csv \
  --out data/processed/fetch_index.json
```

Writes ligand SMILES files, FASTA files, PDB files, a molecule CSV
(`SMILES, chembl_id, pIC50`), and a JSON index of all fetched paths.

### `qsar-descriptors` — Featurise a SMILES CSV

```bash
caddack qsar-descriptors \
  --csv data/raw/mols.csv \
  --smiles-col SMILES \
  --radius 2 --bits 2048 \
  --out data/processed/features.parquet
```

Produces a parquet (or CSV fallback) with 6 physicochemical columns + 2048 ECFP bit
columns + `SMILES_canonical`. Rows with unparseable SMILES get `__error = "invalid_smiles"`.

### `train-gnn` — Train a ligand GNN

```bash
caddack train-gnn \
  --csv data/raw/mols.csv \
  --smiles-col SMILES \
  --target pIC50 \
  --task regression \
  --model gine \
  --epochs 30 \
  --outdir models/gnn
```

Trains `MolecularGCN` (`--model gcn`) or `MolecularGINE` (`--model gine`, default).
Saves `model.pt` + `metrics.json`.

### `train-fusion` — Train the two-tower Bayesian affinity model

Requires a **PDBbind-style dataset**: a root directory where each subdirectory is named by
PDB ID and contains `<pdb_id>_protein.pdb` and `<pdb_id>_ligand.sdf`, plus an index CSV.

```bash
caddack train-fusion \
  --root /path/to/pdbbind/refined-set \
  --index-csv /path/to/affinities.csv \
  --target affinity \
  --cutoff 6.0 \
  --hidden 128 \
  --gine-layers 3 \
  --geo-interactions 3 \
  --kl-warmup 10 \
  --epochs 100 \
  --batch-size 16 \
  --split scaffold \
  --mc-samples-eval 30 \
  --outdir models/fusion
```

Outputs `model.pt`, `metrics.json` (MAE, R², n_train, n_test), and `config.json`
(all architecture hyperparams). Prints a JSON summary to stdout on completion.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--cutoff` | 6.0 | Pocket extraction radius in Å |
| `--hidden` | 128 | Hidden channels for both towers |
| `--kl-warmup` | 10 | Epochs over which KL weight ramps from 0 → `--kl-weight` |
| `--kl-weight` | 1.0 | Final KL weight in the ELBO |
| `--prior-sigma` | 1.0 | Std of the weight prior `N(0, σ²)` |
| `--mc-samples-eval` | 30 | MC samples for epistemic + aleatoric uncertainty at eval |
| `--split` | scaffold | `scaffold` (Murcko) or `random` |
| `--no-aleatoric` | — | Use MSE instead of heteroscedastic NLL |

### `benchmark-fusion` — Speed / calibration / accuracy benchmarks

```bash
python scripts/benchmark_fusion.py \
  --n-complexes 64 \
  --epochs 20 \
  --hidden 64 \
  --out results/bench.json
```

Generates synthetic complexes, runs three benchmark sections:

1. **Speed** — forward pass latency (batch=8) and MC sampling wall time for n_samples in
   {5, 10, 30, 50}.
2. **Calibration (ECE)** — after training, runs 30 MC samples per test complex and checks
   whether predicted Gaussian intervals match empirical coverage. ECE ≈ 0 means the model's
   stated confidence intervals are well-calibrated.
3. **Accuracy vs baselines** — MAE and R² for three models on the same test split:
   - Mean predictor (floor)
   - GINE-only (2D message passing, no geometry, 20 epochs)
   - FusionAffinityNet (full two-tower)

---

## Changes from the original `gpt` branch

### Bug fixes

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

### Improvements

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

### New: Two-tower Bayesian fusion model

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

## Quickstart: fetch → fusion training

```bash
# 1. Fetch a protein structure and ligands
caddack fetch \
  --chembl-target CHEMBL203 \
  --pdb 4EK3 \
  --emit-mols-csv data/raw/mols.csv \
  --out data/processed/fetch_index.json

# 2. Train the two-tower Bayesian affinity model
#    (requires a PDBbind-style dataset with .pdb + .sdf pairs)
caddack train-fusion \
  --root /path/to/pdbbind \
  --index-csv /path/to/affinities.csv \
  --target affinity \
  --outdir models/fusion

# 3. Run benchmarks on synthetic data to verify the installation
python scripts/benchmark_fusion.py --n-complexes 64 --epochs 20
```

---

## Running tests

```bash
# core tests (no RDKit / torch needed)
python -m pytest tests/ -q

# full suite (RDKit + torch + pyg required)
pip install -e ".[gnn,dev]"
python -m pytest tests/ -q
```
