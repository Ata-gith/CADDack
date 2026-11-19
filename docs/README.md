# The browser demo

A static page that predicts aqueous solubility and says when it has no business
answering. Live at **https://ata-gith.github.io/CADDack/**.

Everything runs client-side. RDKit is compiled to WebAssembly and the random
forest is walked in JavaScript, so nothing is uploaded and there is no server to
run. GitHub Pages serves this directory as-is.

## Files

| File | What it is |
|---|---|
| `index.html` | The page: layout, styling, presets |
| `app.js` | Featurisation, forest evaluation, Tanimoto, rendering |
| `model.json` | Trained forest + training fingerprints, built by `scripts/build_demo_model.py` |
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
