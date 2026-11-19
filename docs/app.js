/* ---------------------------------------------------------------------------
   CADDack solubility demo.

   Everything here runs in the browser: RDKit (WASM) computes the same features
   the Python package computes, and the random forest is evaluated tree by tree
   so the spread across trees gives a genuine uncertainty estimate.

   Feature parity with the Python training pipeline is load-bearing. Two details
   matter and are verified against Python in tests/parity:
     - Morgan fingerprints need `useChirality: true`. RDKit.js silently ignores
       `includeChirality` (the Python spelling), which yields different bits.
     - The feature vector is [6 descriptors..., 1024 ECFP bits] in that order.
   --------------------------------------------------------------------------- */

const MODEL_URL = "model.json";
const N_NEIGHBOURS = 4;

let RDKit = null;
let M = null;            // decoded model bundle
let POPCOUNT = null;     // byte -> set bits

// --- decoding ---------------------------------------------------------------

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const asInt16 = (b) => new Int16Array(b64ToBytes(b).buffer);
const asInt32 = (b) => new Int32Array(b64ToBytes(b).buffer);
const asFloat32 = (b) => new Float32Array(b64ToBytes(b).buffer);

async function loadModel() {
  const bundle = await (await fetch(MODEL_URL)).json();
  const f = bundle.forest;
  POPCOUNT = new Uint8Array(256);
  for (let i = 0; i < 256; i++) POPCOUNT[i] = (i & 1) + POPCOUNT[i >> 1];
  return {
    ...bundle,
    forest: {
      nTrees: f.n_trees,
      offsets: f.offsets,
      feature: asInt16(f.feature),
      threshold: asFloat32(f.threshold),
      left: asInt32(f.left),
      right: asInt32(f.right),
      value: asFloat32(f.value),
    },
    train: {
      ...bundle.train,
      fps: b64ToBytes(bundle.train.fps),
      logS: asFloat32(bundle.train.logS),
      bytesPerMol: bundle.train.n_bits / 8,
    },
  };
}

// --- featurisation ----------------------------------------------------------

/** Keep the largest fragment, mirroring `strip_salts` in caddack.qsar.descriptors.

    The Python pipeline strips counter-ions before featurising, so the browser has
    to as well: feeding "[Na+].[Cl-]" whole produces a different fingerprint from
    the one the model was trained against, and the prediction silently drifts.
    Ties go to the first fragment, matching Python's strictly-greater comparison. */
function stripSalts(smiles) {
  if (!smiles.includes(".")) return smiles;
  let best = null, bestAtoms = -1;
  for (const frag of smiles.split(".")) {
    const m = RDKit.get_mol(frag);
    if (!m) continue;
    if (m.is_valid()) {
      const n = JSON.parse(m.get_descriptors()).NumHeavyAtoms;
      if (n > bestAtoms) { bestAtoms = n; best = frag; }
    }
    m.delete();
  }
  return best ?? smiles;
}

/** Build the model's input vector from a SMILES, or null if RDKit rejects it. */
function featurize(smiles) {
  const parent = stripSalts(smiles);
  const mol = RDKit.get_mol(parent);
  if (!mol || !mol.is_valid()) { mol?.delete(); return null; }

  const d = JSON.parse(mol.get_descriptors());
  // Names differ between the Python and JS RDKit bindings; this mapping is the
  // one verified to agree numerically.
  const NAMES = { MolWt: "amw", LogP: "CrippenClogP", TPSA: "tpsa",
                  NumHBD: "NumHBD", NumHBA: "NumHBA", NumRotBonds: "NumRotatableBonds" };
  const descriptors = M.feature_order.map((k) => d[NAMES[k]]);

  const fpStr = mol.get_morgan_fp(
    JSON.stringify({ radius: 2, nBits: M.n_bits, useChirality: true }));
  const svg = mol.get_svg_with_highlights(JSON.stringify({ width: 260, height: 190 }));
  const canonical = mol.get_smiles();
  mol.delete();

  const x = new Float32Array(descriptors.length + M.n_bits);
  descriptors.forEach((v, i) => { x[i] = v; });
  for (let i = 0; i < M.n_bits; i++) {
    if (fpStr[i] === "1") x[descriptors.length + i] = 1;
  }

  // Packed form, for Tanimoto against the packed training fingerprints.
  const packed = new Uint8Array(M.n_bits / 8);
  for (let i = 0; i < M.n_bits; i++) {
    if (fpStr[i] === "1") packed[i >> 3] |= 0x80 >> (i & 7);
  }

  const named = {};
  M.feature_order.forEach((k, i) => { named[k] = descriptors[i]; });
  return { x, packed, svg, canonical, descriptors: named };
}

// --- inference --------------------------------------------------------------

/** Every tree's prediction. Their spread is the epistemic uncertainty. */
function predictPerTree(x) {
  const { nTrees, offsets, feature, threshold, left, right, value } = M.forest;
  const out = new Float64Array(nTrees);
  for (let t = 0; t < nTrees; t++) {
    const base = offsets[t];
    let node = 0;
    while (left[base + node] !== -1) {
      node = x[feature[base + node]] <= threshold[base + node]
        ? left[base + node] : right[base + node];
    }
    out[t] = value[base + node];
  }
  return out;
}

function meanSd(a) {
  const mean = a.reduce((s, v) => s + v, 0) / a.length;
  const varr = a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length;
  return { mean, sd: Math.sqrt(varr) };
}

/** Tanimoto of a packed query against every packed training fingerprint. */
function similarities(packed) {
  const { fps, n, bytesPerMol } = M.train;
  const sims = new Float32Array(n);
  let qBits = 0;
  for (let b = 0; b < bytesPerMol; b++) qBits += POPCOUNT[packed[b]];
  for (let i = 0; i < n; i++) {
    const off = i * bytesPerMol;
    let inter = 0, tBits = 0;
    for (let b = 0; b < bytesPerMol; b++) {
      const tb = fps[off + b];
      inter += POPCOUNT[packed[b] & tb];
      tBits += POPCOUNT[tb];
    }
    const union = qBits + tBits - inter;
    sims[i] = union > 0 ? inter / union : 0;
  }
  return sims;
}

function topNeighbours(sims, k) {
  return Array.from(sims, (s, i) => [s, i])
    .sort((a, b) => b[0] - a[0]).slice(0, k)
    .map(([s, i]) => ({ similarity: s, smiles: M.train.smiles[i], logS: M.train.logS[i] }));
}

// --- rendering --------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function drawSmiles(smiles, w = 150, h = 110) {
  const mol = RDKit.get_mol(smiles);
  if (!mol || !mol.is_valid()) { mol?.delete(); return ""; }
  const svg = mol.get_svg_with_highlights(JSON.stringify({ width: w, height: h }));
  mol.delete();
  return svg;
}

/** Solubility in mol/L is log-scaled; show something a person can picture. */
function plainLanguage(logS) {
  const gPerL = Math.pow(10, logS) * 200;  // ~200 g/mol stands in for a typical drug
  if (logS > -1) return "very soluble — dissolves readily in water";
  if (logS > -2) return "soluble";
  if (logS > -4) return "moderately soluble";
  if (logS > -6) return `poorly soluble — roughly ${gPerL.toFixed(2)} g/L`;
  return "practically insoluble in water";
}

function inTrainingSet(canonical) {
  if (!M._trainSet) M._trainSet = new Set(M.train.smiles);
  return M._trainSet.has(canonical);
}

function verdict(sim, memorised) {
  if (memorised) return { label: "This one is in the training set", cls: "good",
    note: "The model saw this exact molecule while training, so it is recalling rather than generalising. Useful as a sanity check, not as evidence it works." };
  const t = M.ad.threshold;
  if (sim >= t + 0.25) return { label: "Well inside what I know", cls: "good", note: "There are close analogues in the training set, so this prediction is on solid ground." };
  if (sim >= t) return { label: "Inside my domain, just", cls: "ok", note: "Reasonably similar to training molecules. Treat the interval as real." };
  if (sim >= t * 0.6) return { label: "Edge of what I know", cls: "warn", note: "Nothing closely similar was in training. The number is a guess with a wide interval." };
  return { label: "I don't know this chemistry", cls: "bad", note: "Nothing in the training set resembles this. Do not trust the number — this is exactly where models fail silently." };
}

function render(smiles) {
  const err = $("error");
  const f = featurize(smiles);
  if (!f) {
    err.textContent = `Not a valid SMILES: ${smiles}`;
    err.hidden = false;
    $("results").hidden = true;
    return;
  }
  err.hidden = true;

  const trees = predictPerTree(f.x);
  const { mean, sd } = meanSd(trees);
  const sims = similarities(f.packed);
  const maxSim = sims.reduce((a, b) => Math.max(a, b), 0);
  const memorised = inTrainingSet(f.canonical);
  const v = verdict(maxSim, memorised);

  $("depiction").innerHTML = f.svg;
  $("canonical").textContent = f.canonical;
  $("value").textContent = mean.toFixed(2);
  $("interval").textContent = `± ${(1.96 * sd).toFixed(2)}`;
  $("plain").textContent = plainLanguage(mean);

  // Uncertainty bar, drawn on the data's own range.
  const LO = -11, HI = 2;
  const pct = (x) => Math.max(0, Math.min(100, ((x - LO) / (HI - LO)) * 100));
  $("barFill").style.left = `${pct(mean - 1.96 * sd)}%`;
  $("barFill").style.width = `${pct(mean + 1.96 * sd) - pct(mean - 1.96 * sd)}%`;
  $("barMark").style.left = `${pct(mean)}%`;

  const gauge = $("gauge");
  gauge.className = `gauge ${v.cls}`;
  $("gaugeLabel").textContent = v.label;
  $("gaugeNote").textContent = v.note;
  const heavy = f.descriptors.MolWt / 14;
  const saturated = !memorised && maxSim > 0.95 && f.descriptors.NumRotBonds > 12;
  $("caveat").hidden = !saturated;
  $("simValue").textContent = maxSim.toFixed(2);
  $("simThreshold").textContent = M.ad.threshold.toFixed(2);
  $("gaugeFill").style.width = `${Math.min(100, maxSim * 100)}%`;
  $("spread").textContent = sd.toFixed(2);

  $("neighbours").innerHTML = topNeighbours(sims, N_NEIGHBOURS).map((nb) => `
    <figure class="nb">
      <div class="nbsvg">${drawSmiles(nb.smiles)}</div>
      <figcaption>
        <strong>${(nb.similarity * 100).toFixed(0)}%</strong> similar
        <span>measured ${nb.logS.toFixed(2)}</span>
      </figcaption>
    </figure>`).join("");

  $("results").hidden = false;
}

// --- wiring -----------------------------------------------------------------

function submit(value) {
  const smiles = (value ?? $("smiles").value).trim();
  if (!smiles) return;
  $("smiles").value = smiles;
  render(smiles);
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    RDKit = await initRDKitModule();
    M = await loadModel();
  } catch (e) {
    $("error").textContent = `Could not start: ${e}`;
    $("error").hidden = false;
    return;
  }

  $("trainN").textContent = M.train.n.toLocaleString();
  $("r2").textContent = M.metrics.r2.toFixed(2);
  $("mae").textContent = M.metrics.mae.toFixed(2);

  $("loading").hidden = true;
  $("app").hidden = false;

  $("form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  document.querySelectorAll("[data-smiles]").forEach((b) =>
    b.addEventListener("click", () => submit(b.dataset.smiles)));

  submit("CC(=O)Oc1ccccc1C(=O)O");  // aspirin: familiar, and genuinely not in the training set
});
