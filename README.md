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

CADDack/
configs/ # YAML configs for runs
data/ # raw/processed data (gitignored except keepers)
docs/ # notes, design docs
examples/ # tiny runnable examples
notebooks/ # analysis notebooks
scripts/ # CLI entrypoints that call into src/caddack/*
src/caddack/ # package code (data, qsar, gnn, design, docking)
tests/ # pytest suites
README.md
LICENSE
pyproject.toml
environment.yml
.gitignore

---

## Install

### Conda
```bash
conda env create -f environment.yml
conda activate caddack
pip install -e .