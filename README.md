# CADDack

A modular toolkit for computer-aided drug design: data fetching, QSAR featurisation,
2D message-passing GNNs, and a two-tower Bayesian model that predicts protein–ligand
binding affinity with calibrated uncertainty.

```
src/caddack/
├── qsar/      SMILES featurisation (descriptors + ECFP), scaffold splitting
├── gnn/       molecular graphs, GCN/GINE models, Bayesian fusion model, training
└── fetch/     ChEMBL / UniProt / RCSB PDB downloads
```

## Install

```bash
pip install -e .            # core (numpy, pandas, scikit-learn)
pip install -e ".[gnn]"     # + torch, torch-geometric, rdkit
```

Optional dependencies are lazy: every module **imports** without them and raises a clear
install hint only at call time.

## Modules

### `caddack.qsar` — featurisation and splitting

| Module | What it does |
|---|---|
| `descriptors.py` | SMILES → features. Salt stripping, canonicalisation, 6 physicochemical descriptors (`MolWt`, `LogP`, `TPSA`, `NumHBD`, `NumHBA`, `NumRotBonds`) and ECFP fingerprint bits. `featurize_dataframe()` featurises a whole CSV, flagging invalid rows instead of raising. |
| `split.py` | `scaffold_split()` — group molecules by Murcko scaffold and hold out whole scaffold groups, so test compounds are chemically novel. Much harder (and more honest) than a random split. |
| `train_qsar.py` | Random-forest baselines on the featurised data. |

### `caddack.gnn` — graph neural networks

| Module | What it does |
|---|---|
| `datasets.py` | SMILES → graph tensors. 7-dim atom features, 7-dim bond features, bidirectional edges. Unparseable SMILES are skipped with a warning rather than aborting a run. |
| `models.py` | `MolecularGCN` (fast baseline), `MolecularGINE` (uses bond features — preferred for molecules), and `FusionAffinityNet` (the two-tower affinity model). |
| `bayes.py` | Bayes-by-Backprop layers and the ELBO objective. Weights are Gaussian posteriors sampled via the reparameterisation trick; the loss is a per-example NLL plus a KL term scaled by the dataset size. |
| `geometry.py` | PDB/mol2 parsing and binding-pocket extraction — turns a protein + ligand pair into a 3D `GeometryRecord`. |
| `train.py` | `train_from_csv()` for ligand-only models, `train_fusion_from_complexes()` for the fusion model. |

All models use a factory pattern (`Model.build(...)`) so the modules stay importable
without torch.

### The fusion model

`FusionAffinityNet` combines two encoders and a Bayesian head:

- **Tower 1 (2D)** — a GINE network over the ligand graph.
- **Tower 2 (3D)** — a geometry network over pocket + ligand atoms with radial-basis
  distance features.
- **Bayesian head** — outputs a mean affinity and a log-variance, giving predictive
  uncertainty split into *epistemic* (model, shrinks with more data) and *aleatoric*
  (measurement noise, irreducible).

The mathematics — what data enters, the objective, and how it is minimised — is written up
in [`docs/elbo_math.pdf`](docs/elbo_math.pdf) (source: `docs/elbo_math.tex`).

## CLI

```bash
caddack <subcommand> --help
```

| Subcommand | Purpose |
|---|---|
| `fetch` | Download ligands/sequences/structures from ChEMBL, UniProt, RCSB PDB. |
| `qsar-descriptors` | Featurise a CSV of SMILES → parquet/CSV. |
| `qsar-train` | Train a random-forest QSAR baseline. |
| `train-gnn` | Train `MolecularGCN`/`MolecularGINE` from a SMILES CSV. |
| `train-fusion` | Train the two-tower Bayesian affinity model. |

## Benchmarks

### 2D GNNs on MoleculeNet

```bash
python scripts/benchmark_molnet.py --download \
  --data-dir /tmp/molnet --outdir /tmp/molnet_runs --epochs 40 --model gine
```

Random 80/20 split, 40 epochs, hidden=128, 3 layers:

| Dataset | Task | N | GINE | GCN |
|---|---|---|---|---|
| ESOL | regression | 1,128 | MAE 0.90 / R² 0.74 | MAE 1.13 / R² 0.59 |
| FreeSolv | regression | 642 | MAE 1.96 / R² 0.57 | MAE 2.22 / R² 0.34 |
| BBBP | classification | 2,039 | AUC 0.834 | AUC 0.777 |
| BACE | classification | 1,513 | AUC 0.721 | AUC 0.712 |

GINE wins on every dataset, as expected — it consumes bond features that GCN ignores.

### Fusion model on PDBbind

```bash
# one-time: download (~2 GB) and cache the parsed complexes
python scripts/benchmark_pdbbind.py --data-dir /tmp/pdbbind --download --build

# CASF-style: train on refined-minus-core, test on the core set
python scripts/benchmark_pdbbind.py --data-dir /tmp/pdbbind --refined-only \
  --split core --core-csv /tmp/pdbbind/pdbbind_core_df.csv.gz --epochs 15 --prefix refined
```

Affinity in p*K* units (−log *K*d/*K*i):

| Training data | Split | Train/Test | MAE | R² | Pearson r | Coverage 1σ/2σ |
|---|---|---|---|---|---|---|
| Refined, 15 ep | core holdout | 3,511 / 188 | 1.28 | 0.47 | 0.69 | 64% / 94% |
| Refined, 30 ep | core holdout | 3,511 / 188 | 1.34 | 0.42 | 0.73 | 66% / 96% |
| General, 10 ep | core holdout | 6,000 / 193 | 1.34 | 0.44 | 0.68 | 67% / 96% |
| Refined, 20 ep | **scaffold** | 2,959 / 740 | 1.32 | 0.31 | 0.57 | 67% / 94% |
| *mean predictor* | — | — | 1.82 | 0.00 | — | — |

A few notes on reading them:

- Every run beats the mean-predictor baseline, and the uncertainty estimates are
  reasonably calibrated (ideal coverage is 68% / 95%).
- The core-holdout numbers are the optimistic ones — the refined and core sets share many
  scaffolds. The scaffold split is the stricter test, since those test ligands are
  chemically new to the model.
- More epochs and more data both plateau quickly, so the limit is not training volume.

#### Two settings that mattered

Both were found on the scaffold split. Standardisation is on by default; tempering is opt-in
via `--kl-weight`.

| Scaffold split, 20 epochs | MAE | RMSE | R² | Pearson r | Coverage 1σ/2σ |
|---|---|---|---|---|---|
| raw target, `kl_weight=1.0` | 1.59 | 2.04 | −0.04 | 0.60 | 54% / 84% |
| z-scored target | 1.48 | 1.78 | 0.21 | 0.58 | 68% / 96% |
| z-scored + `--kl-weight 0.01` | **1.32** | **1.66** | **0.31** | 0.57 | 67% / 94% |

**Target standardisation.** Training on raw affinities left a systematic +1.05 p*K* offset,
since the Bayesian head is regularised toward a zero-mean prior while being asked to emit
values centred near 6.4. Standardisation uses train-set statistics only and is undone at
prediction time (`y_mean`/`y_std` live in `config.json`).

**KL tempering.** The Bayesian head carries ~99k weight posteriors but trains on ~3k
complexes, so the correctly-scaled ELBO is roughly 98% KL and 2% data fit. Lowering
`--kl-weight` to 0.01–0.1 rebalances it; most of the benefit is simply leaving 1.0.

Worth noting what these did *not* change: Pearson r stays near 0.58 throughout. Both
settings fix calibration — the offset and the scale — rather than the ranking. R² now sits
close to r², so there is little left to gain from that direction, and the remaining limit
looks like a representation one. The geometry tower currently sees only atomic number and
interatomic distance, with no explicit hydrogen bonds, hydrophobic contacts, or residue
identity.

#### How this compares to published models

For context, some published results on the CASF-2016 benchmark:

| Model | Pearson r | RMSE |
|---|---|---|
| AutoDock Vina | 0.57 | — |
| Classical scoring functions (X-Score et al.) | ≲0.61 | — |
| Pafnucy | 0.78 | 1.42 |
| OnionNet / DeepAtom | 0.81 | 1.28 / 1.32 |
| OnionNet-2 | 0.86 | 1.16 |

Our most comparable figure is the core-holdout r ≈ 0.73 (RMSE ≈ 1.68). That sits above
classical scoring functions and below the tuned deep models, which is a fair reflection of
where this project is: an untuned model trained for 15–30 CPU epochs, with a deliberately
simple feature set, against methods that have had far more engineering invested in them.

Two caveats, in both directions. Ours is not a like-for-like comparison — we test on the
~190-complex DeepChem core set rather than the official 285-complex CASF-2016 set, so
please treat the table as rough context rather than a ranking. And recent work reports that
[CASF-2016 overlaps heavily with PDBbind training data](https://academic.oup.com/bioinformatics/article/41/2/btaf040/7985708),
so [published figures may be inflated by train–test leakage](https://www.nature.com/articles/s42256-025-01124-5).
That applies to our core-holdout number too; it is a reason to read all of these
cautiously, not a claim about the gap.

### Synthetic speed / calibration

```bash
python scripts/benchmark_fusion.py --n-complexes 64 --epochs 20 --out results/bench.json
```

Uses synthetic complexes, so it validates throughput, the uncertainty decomposition and
the training loop — **not** predictive quality.

## Quickstart

```bash
# 1. fetch a target's ligands
caddack fetch --chembl-target CHEMBL203 --min-n 100 --emit-mols-csv data/egfr.csv

# 2. featurise and train a QSAR baseline
caddack qsar-descriptors --csv data/egfr.csv --smiles-col SMILES --out data/feats.parquet
caddack qsar-train --parquet data/feats.parquet --target pIC50 --split scaffold

# 3. train a ligand GNN
caddack train-gnn --csv data/egfr.csv --smiles-col SMILES --target-col pIC50 \
  --model gine --task regression
```

## Tests

```bash
pytest tests/ -q          # optional-dependency tests skip automatically
```

## Further reading

- [`docs/elbo_math.pdf`](docs/elbo_math.pdf) — the ELBO objective end to end.
- [`docs/caddack_gnn_math.pdf`](docs/caddack_gnn_math.pdf) — full mathematics of every layer.
- [`CHANGELOG.md`](CHANGELOG.md) — change history.
