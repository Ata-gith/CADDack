# CADDack

A computational drug-design toolkit: fetch molecules, featurise them, train models
that predict binding affinity, and generate docked poses — with honest uncertainty
and honest evaluation throughout.

**[→ Try the live demo](https://ata-gith.github.io/CADDack/)** — predict a molecule's
water solubility in your browser and watch the model admit when it's out of its depth.
No install, nothing uploaded.

---

## What it does

```
src/caddack/
├── qsar/      SMILES → features, scaffold splitting, curation, applicability domain
├── gnn/       molecular graphs, GCN/GINE, a Bayesian two-tower affinity model
├── docking/   AutoDock Vina pose generation
└── fetch/     ChEMBL / UniProt / RCSB downloads
```

Four things it tries to do properly:

- **Honest splits.** Train/test splits hold out whole Bemis–Murcko scaffolds, so test
  molecules are structurally novel rather than cousins of the training set.
- **Honest uncertainty.** The affinity model is Bayesian and separates *epistemic*
  uncertainty (which shrinks with more data) from *aleatoric* noise (which doesn't).
- **Honest scope.** An applicability domain flags predictions on chemistry the model
  never saw, instead of extrapolating confidently.
- **Honest reporting.** Metrics come from repeated splits with a standard deviation,
  and every number records which split produced it.

## Install

```bash
pip install -e .              # core
pip install -e ".[gnn]"       # + torch, torch-geometric, rdkit
pip install -e ".[docking]"   # + vina, meeko, rdkit
```

Or `conda env create -f environment.yml` — conda-forge ships RDKit, PyTorch and Vina
as binaries, which avoids building Vina from source. Pinned versions in
`requirements.lock`.

Optional dependencies are lazy: every module imports without them and raises a clear
install hint only when you actually call something that needs them.

## Quickstart

```bash
# featurise a CSV of SMILES, then train a baseline with repeated scaffold CV
caddack qsar-descriptors --csv mols.csv --smiles-col SMILES --out feats.parquet
caddack qsar-train --parquet feats.parquet --target pIC50 --n-repeats 5

# dock a ligand into a receptor
caddack dock --receptor protein.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O" \
             --reference-ligand crystal_ligand.sdf
```

```python
from caddack.qsar.applicability import SimilarityAD

ad = SimilarityAD.fit(train_fingerprints)
ad.in_domain(query_fingerprints)      # False => don't trust the prediction
```

Run `caddack --help` for all subcommands: `fetch`, `qsar-descriptors`, `qsar-train`,
`train-gnn`, `train-fusion`, `dock`.

## How well does it work?

**Solubility (ESOL, 1,128 molecules, scaffold split)** — the live demo's model:

| R² | MAE | Pearson *r* |
|---|---|---|
| 0.78 | 0.74 log units | 0.90 |

**Binding affinity (PDBbind, p*K* units)** — the Bayesian two-tower model:

| Training set | Split | MAE | R² | Pearson *r* | Coverage 1σ / 2σ |
|---|---|---|---|---|---|
| Refined | core holdout | 1.28 | 0.47 | 0.69 | 64% / 94% |
| Refined | **scaffold** | 1.32 | 0.31 | 0.57 | 67% / 94% |
| *predict the mean* | — | 1.82 | 0.00 | — | — |

Uncertainty estimates are close to calibrated (ideal is 68% / 95%). For context, on
CASF-style core holdouts classical scoring functions reach Pearson *r* ≈ 0.57–0.61 and
tuned deep models 0.78–0.86; this sits between them, which is a fair reflection of an
untuned model with a deliberately simple feature set. That comparison is rough — ours
uses a different core set — and those benchmarks are
[known to overlap with PDBbind training data](https://www.nature.com/articles/s42256-025-01124-5),
so published figures and ours alike deserve a sceptical read. The scaffold row is the
one to judge it by.

**Re-docking (30 PDBbind complexes)** — fraction reproducing the crystal pose below 2 Å:

| Protocol | Top pose | Any of 9 |
|---|---|---|
| re-embedded ligand, `exhaustiveness=8` | 43% | 70% |
| **crystal conformer, `exhaustiveness=32`** | **60%** | **90%** |

The gap between those rows is mostly a lesson: Vina samples torsions but keeps ring
conformations fixed, so starting from a re-embedded conformer costs real accuracy.

**2D GNNs on MoleculeNet** (random split): GINE reaches ESOL R² 0.74, BBBP AUC 0.834,
BACE AUC 0.721 — beating GCN on every dataset, as expected, since it uses bond features.

Reproduce any of these with the scripts in `scripts/`.

## Tests

```bash
pip install -e ".[gnn]" && pytest
```

Tests requiring optional dependencies skip cleanly without them. CI builds a wheel and
tests *that*, with and without the `[gnn]` extra, so the installed package is what gets
checked rather than the working tree.

## Further reading

- **[`docs/README.md`](docs/README.md)** — how the browser demo works, and why feature
  parity between Python and RDKit.js is load-bearing.
- **[`docs/elbo_math.pdf`](docs/elbo_math.pdf)** — the ELBO objective end to end: what
  enters it and how it's minimised.
- **[`docs/caddack_gnn_math.pdf`](docs/caddack_gnn_math.pdf)** — full layer mathematics
  for both towers.
- **[`docs/autodock_tutorial.pdf`](docs/autodock_tutorial.pdf)** — a practical docking
  guide, including nine pitfalls found while building this.
- **[`CHANGELOG.md`](CHANGELOG.md)** — development history.

## Licence

MIT.
