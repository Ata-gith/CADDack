# The browser demos

Two static pages, live at **https://ata-gith.github.io/CADDack/**.

| Page | Model | Runs where |
|---|---|---|
| `index.html` | RandomForest on ECFP4 + descriptors, trained on ESOL | **live in the browser** |
| `fusion.html` | `FusionAffinityNet` — two-tower Bayesian GNN, trained on PDBbind | precomputed |

They are different models answering different questions, and the pages say so.
The solubility page takes any SMILES and predicts live. The affinity page cannot:
`FusionAffinityNet` needs a protein–ligand complex with 3D coordinates, which a
visitor cannot type in, and its Bayesian head needs Monte Carlo sampling over
weight posteriors. So that page explores predictions computed offline by
`scripts/build_fusion_demo.py`, and it uses the room that buys to ask something a
live predictor could not: whether the uncertainty the model reports actually
tracks the errors it makes.

Everything runs client-side. RDKit is compiled to WebAssembly and the random
forest is walked in JavaScript, so nothing is uploaded and there is no server to
run. GitHub Pages serves this directory as-is.

## Files

| File | What it is |
|---|---|
| `index.html` / `app.js` | Solubility page: featurisation, forest evaluation, Tanimoto, rendering |
| `model.json` | Trained forest + training fingerprints, from `scripts/build_demo_model.py` |
| `fusion.html` / `fusion.js` | Affinity page: parity plot, reliability chart, 3D viewer |
| `fusion.json` | Held-out predictions with uncertainty, from `scripts/build_fusion_demo.py` |
| `style.css` | Design tokens and components shared by both pages |
| `3Dmol-min.js` | 3Dmol.js, vendored from npm, for the binding-site viewer |
| `RDKit_minimal.js` / `.wasm` | RDKit's MinimalLib, vendored from npm (see below) |
| `.nojekyll` | Stops Jekyll filtering files and slowing the Pages build |
| `*.tex` / `*.pdf` | The mathematics write-ups, unrelated to the demo |

## Rebuilding the model

```bash
python scripts/build_demo_model.py      # downloads ESOL, writes docs/model.json
```

Trains on [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x) (1,128 molecules)
using CADDack's own featurisation, `scaffold_split` and `SimilarityAD`. Held out
by Bemis–Murcko scaffold, so the test molecules are structurally novel:

| | |
|---|---|
| R² | 0.78 |
| MAE | 0.74 log units |
| Pearson *r* | 0.90 |
| Train / test | 902 / 226 across 268 scaffold groups |

## Rebuilding the affinity predictions

```bash
# one-time: ~2 GB download, then parse ~12,000 complexes
python scripts/benchmark_pdbbind.py --data-dir /path/to/pdbbind --download --build
python scripts/build_fusion_demo.py --data-dir /path/to/pdbbind --epochs 25
```

Trains on the PDBbind v2015 refined set with a Bemis–Murcko scaffold split, so the
test ligands are chemically novel rather than close analogues. Training resumes
from `fusion_demo_ckpt.pt` if interrupted. The export ships per-complex
predictions for the whole test set plus 3D coordinates for a subset chosen to span
the behaviour worth seeing — the most accurate calls, the worst misses, and the
cases where the model was most and least certain.

The pocket is everything within 6 Å of the ligand, so the model never sees the
wider protein.

## Two fixes to the affinity model, and what they bought

**Monte Carlo dropout during sampling.** `predict_with_uncertainty` previously
called `.eval()`, which disabled dropout in the encoders. Across 30 samples only
the Bayesian head's ~198k weights varied; the ~204k encoder parameters produced
an identical embedding every pass. The epistemic term could therefore describe
uncertainty about the final linear map, but never that a complex was unlike
anything in training. It now keeps dropout live (`mc_dropout=True`, the default);
pass `--no-mc-dropout` to reproduce the old behaviour.

**First-neighbour geometry features.** The tower embedded atomic number alone, so
a backbone carbonyl oxygen, a serine hydroxyl and a buried water were one token —
and nothing told it which atoms were ligand and which were protein. Each atom now
also carries its coordination number, the element composition of its bonded
neighbours, their distances, and a ligand flag. Bond length does most of the work:
on PDBbind, oxygens with a single carbon neighbour split 82% below 1.30 Å
(carbonyl, acceptor) and 17% above 1.36 Å (hydroxyl, also a donor). Computed from
coordinates in `neighbor_features`, so existing caches stay valid.

Measured on the same scaffold split:

| | before | after |
|---|---|---|
| 1σ coverage (ideal 68%) | 60.5% | **69.1%** |
| 2σ coverage (ideal 95%) | 91.9% | **96.9%** |
| R² | 0.303 | 0.336 |
| MAE (pK) | 1.255 | 1.228 |
| Pearson *r* | 0.604 | 0.601 |
| Spearman ρ(σ, error) | 0.071 | 0.100 |

Calibration is the clear win and it is what MC dropout was expected to fix: the
intervals were too narrow because a whole source of variance was switched off.

**The accuracy column is not a result.** Training the same architecture on the
same split with five different initialisations gives R² 0.306 ± 0.083 (range
0.176–0.390), MAE 1.238 ± 0.086 and *r* 0.602 ± 0.038. The apparent gains above —
R² +0.033, MAE −0.027 — are a third of a standard deviation, so first-neighbour
features did **not** measurably improve accuracy. Two single runs were compared
when seed noise is several times the effect. Only the calibration change is
demonstrated.

Ranking is unchanged either way, consistent with earlier findings that more data,
more epochs, standardisation and KL tempering all left *r* near 0.58-0.60. That
ceiling looks like a representation limit; explicit interaction features are the
next thing to try.

Per-prediction discrimination is still poor (ρ = 0.10, and the quintile trend is
not monotonic). One split, so read it as indicative.

### Which warning signal is actually worth listening to

Comparing every available signal against the errors the model made, by rank
correlation over the held-out complexes, as shipped in `fusion.json`:

| signal | ρ vs \|error\| |
|---|---|
| epistemic σ — the Bayesian head | **−0.061** |
| aleatoric σ — predicted noise | +0.069 |
| both, as the model reports them | +0.049 |
| distance from training chemistry | +0.091 |
| the last two, rank-averaged | **+0.112** |

Two results worth stating plainly. The **epistemic** term — the thing the Bayesian
machinery exists to produce — is slightly *negative*, so even after its calibration
was fixed it does not rank errors. And **`SimilarityAD`, which ignores the model
entirely**, does better than anything the model reports about itself: the 195 test
ligands outside the domain average 1.38 pK of error against 1.18 for those inside.

Averaging their ranks beats either alone, and that combination is what the page
shows. It is still modest, and the better half of it needs no neural network.

**How stable are these?** The uncertainty estimate is stochastic, so the
comparison was repeated over eight independent Monte Carlo passes of the same
trained model:

| signal | mean ρ | sd | range |
|---|---|---|---|
| epistemic σ | −0.069 | 0.016 | −0.094 … −0.048 |
| aleatoric σ | +0.089 | 0.013 | +0.078 … +0.118 |
| both, as reported | +0.087 | 0.022 | +0.063 … +0.134 |
| distance from training chemistry | +0.093 | **0.004** | +0.089 … +0.098 |
| the two, rank-averaged | +0.128 | 0.017 | +0.101 … +0.159 |

Two orderings hold in **every** pass: epistemic never crosses zero, and the
combination beats both of its parts (8/8 against each). The middle three sit
within noise of one another, so nothing should be read into their order.

The applicability-domain signal is also by far the most *reproducible* — its sd
is a quarter of the others', because it does not depend on Monte Carlo sampling
at all. For deciding whether to trust a prediction, that matters as much as the
correlation itself.

Reproduce with `scripts/rho_stability.py`.

## Updating RDKit

The WASM build is vendored rather than loaded from a CDN, so the page keeps
working if a CDN changes or is unreachable:

```bash
npm pack @rdkit/rdkit
tar xzf rdkit-rdkit-*.tgz
cp package/dist/RDKit_minimal.{js,wasm} docs/
```

**Re-run the parity check afterwards.** See below — an upgrade can change
featurisation silently.

## Why the forest is exported tree-by-tree

The demo's point is an honest uncertainty estimate, and the spread across trees
is one. ONNX Runtime Web returns only the ensemble mean and costs ~10 MB of
runtime to do it; walking the trees in JavaScript is about fifteen lines and
gives per-tree predictions for free.

Trees are flattened into five typed arrays with an offset table and
base64-encoded, so the browser decodes flat buffers instead of a JSON object
graph. Training fingerprints ship bit-packed at 128 bytes per molecule rather
than 8 KB, which is what makes shipping the whole training set practical.

## Feature parity is load-bearing

The browser must build *exactly* the feature vector the Python model was trained
on. If it does not, the page serves confident, wrong numbers with nothing to
indicate anything is amiss.

Three details matter:

1. **Fingerprints need chirality.** RDKit.js spells the flag `useChirality`. It
   silently ignores `includeChirality` — the spelling the Python API uses — and
   returns different bits with no error. Chiral alanine scored a Jaccard of
   0.714 against Python before this was caught.
2. **Feature order** is `[MolWt, LogP, TPSA, NumHBD, NumHBA, NumRotBonds,
   ECFP4_0 … ECFP4_1023]`.
3. **Salts must be stripped to the largest fragment**, as `strip_salts` does in
   Python. Featurising `[Na+].[Cl-]` whole rather than its parent fragment changes
   the prediction — this one actually shipped briefly and was caught by comparing
   against Python on multi-fragment inputs.

`tests/test_demo_parity.py` guards the Python half and the shape of the shipped
bundle. The JavaScript half was verified by running this page in headless
Chromium and comparing against the Python model directly — predictions and
similarities agreed to **0.000000** across a spread of molecules including
chiral, ionic and organometallic inputs. Re-run that comparison after upgrading
RDKit.

## Known limits, on purpose

The applicability-domain check is nearest-neighbour Tanimoto, and it has blind
spots. Long aliphatic chains saturate the fingerprint — a 30-carbon chain sets
the same bits as a 20-carbon one — so the gauge reports a close match it has not
earned. Rather than hide that, the page detects the case and contradicts its own
gauge, because a domain check that never admits its own failure modes is the
thing being argued against.

ESOL is small and skews drug-like. Inorganics, salts and polymers are outside
its scope by construction, which is why the gauge exists.
