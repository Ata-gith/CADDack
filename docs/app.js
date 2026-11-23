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
const N_NEIGHBOURS = 8;

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
  const gPerL = Math.pow(10, logS) * 200;   // ~200 g/mol stands in for a typical drug
  if (logS > -1) return "Dissolves readily in water.";
  if (logS > -2) return "Soluble.";
  if (logS > -4) return "Moderately soluble.";
  if (logS > -6) return `Poorly soluble, roughly ${gPerL.toFixed(2)} g/L.`;
  return "Practically insoluble in water.";
}

function inTrainingSet(canonical) {
  if (!M._trainSet) M._trainSet = new Set(M.train.smiles);
  return M._trainSet.has(canonical);
}

function verdict(sim, memorised) {
  if (memorised) return { cls: "good",
    note: "This exact structure is in the reference set, so the similarity is 1.00 and "
        + "the solubility estimate is a lookup rather than a prediction." };
  const t = M.ad.threshold;
  if (sim >= t + 0.25) return { cls: "good",
    note: "Close analogues are present, so the estimate rests on molecules the model "
        + "has actually seen." };
  if (sim >= t) return { cls: "ok",
    note: "Similar enough to the reference set for the interval to be meaningful." };
  if (sim >= t * 0.6) return { cls: "warn",
    note: "Nothing closely similar in the reference set. The interval is wide and the "
        + "number is an extrapolation." };
  return { cls: "bad",
    note: "No reference molecule resembles this one. The solubility estimate is "
        + "extrapolation and should not be relied on." };
}

function render(smiles) {
  const err = $("error");
  const f = featurize(smiles);
  if (!f) {
    err.textContent = `Not a valid SMILES: ${smiles}`;
    err.hidden = false;
    $("results").hidden = true;
    $("predictPanel").hidden = true;
    return;
  }
  err.hidden = true;

  const sims = similarities(f.packed);
  const maxSim = sims.reduce((a, b) => Math.max(a, b), 0);
  const memorised = inTrainingSet(f.canonical);
  const v = verdict(maxSim, memorised);
  const above = sims.reduce((n, s) => n + (s >= M.ad.threshold ? 1 : 0), 0);

  // --- the search result ---
  $("depiction").innerHTML = f.svg;
  $("canonical").textContent = f.canonical;
  $("simValue").textContent = maxSim.toFixed(2);
  $("simCount").textContent = above.toLocaleString();
  $("simTotal").textContent = M.train.n.toLocaleString();
  $("simThreshold").textContent = M.ad.threshold.toFixed(2);
  $("gaugeFill").style.width = `${Math.min(100, maxSim * 100)}%`;
  $("queryBlock").className = `query ${v.cls}`;

  const nbs = topNeighbours(sims, N_NEIGHBOURS);
  $("simSummary").textContent = memorised
    ? "This structure is already in the reference set."
    : above === 0
      ? "Nothing in the reference set clears the similarity threshold."
      : `${above} molecule${above === 1 ? "" : "s"} clear the threshold. The ${nbs.length} closest are shown below.`;

  $("neighbours").innerHTML = nbs.map((nb) => `
    <figure class="nb">
      <div class="nbsvg">${drawSmiles(nb.smiles)}</div>
      <figcaption>
        <strong>${nb.similarity.toFixed(2)}</strong>
        <span>log S ${nb.logS.toFixed(2)}</span>
      </figcaption>
    </figure>`).join("");

  // --- the secondary solubility estimate ---
  const trees = predictPerTree(f.x);
  const { mean, sd } = meanSd(trees);
  $("value").textContent = mean.toFixed(2);
  $("interval").textContent = `± ${(1.96 * sd).toFixed(2)}`;
  $("plain").textContent = plainLanguage(mean);

  const LO = -11, HI = 2;
  const pct = (x) => Math.max(0, Math.min(100, ((x - LO) / (HI - LO)) * 100));
  $("barFill").style.left = `${pct(mean - 1.96 * sd)}%`;
  $("barFill").style.width = `${pct(mean + 1.96 * sd) - pct(mean - 1.96 * sd)}%`;
  $("barMark").style.left = `${pct(mean)}%`;

  $("gauge").className = `flag ${v.cls}`;
  $("gaugeNote").textContent = v.note;
  $("caveat").hidden = !(!memorised && maxSim > 0.95 && f.descriptors.NumRotBonds > 12);

  $("results").hidden = false;
  $("predictPanel").hidden = false;
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

  for (const id of ["trainN", "trainN2"]) {
    const el = $(id);
    if (el) el.textContent = M.train.n.toLocaleString();
  }
  $("r2").textContent = M.metrics.r2.toFixed(2);
  $("mae").textContent = M.metrics.mae.toFixed(2);

  $("loading").hidden = true;
  $("app").hidden = false;

  $("form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  document.querySelectorAll("[data-smiles]").forEach((b) =>
    b.addEventListener("click", () => submit(b.dataset.smiles)));

  submit("CC(=O)Oc1ccccc1C(=O)O");  // aspirin: familiar, and genuinely not in the training set
});
