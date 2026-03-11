# CADDack

Modular, scriptable toolkit for end-to-end computer-aided drug design (CADD). Focus: clean data plumbing, reproducible QSAR, configurable GNNs, inverse molecular design, and docking validation.

---

## Core modules

- **Data integration**: fetch ligands/proteins from public sources (ChEMBL, UniProt, PDB).
- **QSAR**: RDKit descriptors, fingerprints, dataset hygiene.
- **GNNs**: PyTorch Geometric models for regression/classification.
- **Inverse design**: conditional generative modeling over molecular strings (e.g., SELFIES).
- **Docking**: AutoDock Vina / HADDOCK orchestration for structure-based validation.

---

## Repository layout

- CADDack/
- configs/ # YAML configs for runs
- data/ # raw/processed data (gitignored except keepers)
- docs/ # notes, design docs
- examples/ # tiny runnable examples
- notebooks/ # analysis notebooks
- scripts/ # CLI entrypoints that call into src/caddack/*
- src/caddack/ # package code (data, qsar, gnn, design, docking)
- tests/ # pytest suites
- README.md
- LICENSE
- pyproject.toml
- environment.yml
- .gitignore

---

## Install

### Conda
```bash
conda env create -f environment.yml
conda activate caddack
pip install -e .
```

## Fetch → GNN training quickstart

Yes—`train-gnn` can be used directly with data emitted by `fetch`.

```bash
# 1) Fetch ligands and emit a CSV with SMILES + pIC50
caddack fetch \
  --chembl-target CHEMBL203 \
  --min-n 200 \
  --emit-mols-csv data/raw/mols.csv \
  --out data/processed/fetch_index.json

# 2) Train a regression GNN on pIC50 from fetch output
caddack train-gnn \
  --csv data/raw/mols.csv \
  --target pIC50 \
  --task regression \
  --model gine \
  --outdir models/gnn/pIC50_reg

# 3) Or train a binary interaction classifier from pIC50 thresholding
#    (e.g., pIC50 >= 6.0 is "active")
caddack train-gnn \
  --csv data/raw/mols.csv \
  --task classification \
  --target pIC50 \
  --positive-threshold 6.0 \
  --model gine \
  --outdir models/gnn/pIC50_cls
```
