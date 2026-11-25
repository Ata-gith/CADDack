/* ---------------------------------------------------------------------------
   Affinity page.

   Two bundles are loaded. docs/affinity.json holds the random forest that ships
   as the affinity model, and leads the page. docs/fusion.json holds the
   two-tower Bayesian graph network that came first, together with the 3D
   coordinates shown in the viewer and the uncertainty decomposition only it
   produces.

   Both were evaluated on the same scaffold split and the same 740 held-out
   complexes, so predictions can be compared complex by complex.

   Neither model runs in the browser. The forest could -- it is a tree ensemble,
   like the one on the similarity page -- but the predictions shown here are the
   held-out ones, which is the honest thing to display.
   --------------------------------------------------------------------------- */

let F = null;          // forest bundle
let N = null;          // network bundle
let byId = {};         // pdb id -> { forest, network }
let viewer = null;
// Mid-grey chosen to read against both the light panel (#fff) and the dark one
// (#1c1c22); the viewer is transparent, so it sits on whichever is active.
const POCKET_GREY = "#8c929c";

const $ = (id) => document.getElementById(id);
const absErr = (p, m) => Math.abs(p - m);

// --- parity plot -------------------------------------------------------------

const PLOT = { w: 520, h: 520, pad: 48, lo: 1, hi: 13 };

function plotXY(v, axis) {
  const { pad, w, h, lo, hi } = PLOT;
  const t = (v - lo) / (hi - lo);
  return axis === "x" ? pad + t * (w - 2 * pad) : h - pad - t * (h - 2 * pad);
}

function drawParity() {
  const { w, h, pad, lo, hi } = PLOT;
  const ticks = [2, 4, 6, 8, 10, 12];
  const withStructure = new Set(Object.keys(N.structures));

  const grid = ticks.map((t) => `
    <line x1="${plotXY(t, "x")}" y1="${pad}" x2="${plotXY(t, "x")}" y2="${h - pad}" class="grid"/>
    <line x1="${pad}" y1="${plotXY(t, "y")}" x2="${w - pad}" y2="${plotXY(t, "y")}" class="grid"/>
    <text x="${plotXY(t, "x")}" y="${h - pad + 16}" class="tick" text-anchor="middle">${t}</text>
    <text x="${pad - 8}" y="${plotXY(t, "y") + 4}" class="tick" text-anchor="end">${t}</text>`).join("");

  const pts = F.complexes
    .slice()
    .sort((a, b) => withStructure.has(a.id) - withStructure.has(b.id))
    .map((r) => `<circle cx="${plotXY(r.measured, "x").toFixed(1)}"
                         cy="${plotXY(r.pred, "y").toFixed(1)}"
                         r="${(2.4 + r.spread * 1.4).toFixed(1)}"
                         class="pt${withStructure.has(r.id) ? " has3d" : ""}"
                         data-id="${r.id}"><title>${r.id}: measured ${r.measured}, predicted ${r.pred}</title></circle>`)
    .join("");

  $("parity").innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted versus measured affinity">
      ${grid}
      <line x1="${plotXY(lo, "x")}" y1="${plotXY(lo, "y")}"
            x2="${plotXY(hi, "x")}" y2="${plotXY(hi, "y")}" class="diag"/>
      ${pts}
      <text x="${w / 2}" y="${h - 8}" class="axis" text-anchor="middle">measured affinity (pK)</text>
      <text x="14" y="${h / 2}" class="axis" text-anchor="middle"
            transform="rotate(-90 14 ${h / 2})">predicted affinity (pK)</text>
    </svg>`;

  $("parity").querySelectorAll("circle").forEach((c) =>
    c.addEventListener("click", () => select(c.dataset.id)));
}

// --- head to head ------------------------------------------------------------

function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx, b = ys[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return num / Math.sqrt(dx * dy);
}

function stats(records) {
  const pred = records.map((r) => r.pred);
  const meas = records.map((r) => r.measured);
  const err = records.map((r) => absErr(r.pred, r.measured));
  const mean = meas.reduce((s, v) => s + v, 0) / meas.length;
  const ssTot = meas.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ssRes = records.reduce((s, r) => s + (r.pred - r.measured) ** 2, 0);
  return {
    r: pearson(pred, meas),
    mae: err.reduce((s, v) => s + v, 0) / err.length,
    r2: 1 - ssRes / ssTot,
  };
}

function drawHeadToHead() {
  const shared = Object.keys(byId).filter((id) => byId[id].forest && byId[id].network);
  const f = stats(shared.map((id) => byId[id].forest));
  const n = stats(shared.map((id) => byId[id].network));

  const rows = [
    [`Random forest, ${F.feature_names.length} features`, f, true],
    ["Two-tower network", n, false],
  ];
  const best = Math.max(f.r, n.r);

  $("headToHead").innerHTML = rows.map(([label, s, lead]) => `
    <div class="h2hrow${lead ? " lead" : ""}">
      <span class="h2hname">${label}</span>
      <span class="h2hbar"><i style="width:${((s.r / best) * 100).toFixed(0)}%"></i></span>
      <span class="h2hval">${s.r >= 0 ? "+" : ""}${s.r.toFixed(3)}</span>
      <span class="h2hval">${s.mae.toFixed(3)}</span>
      <span class="h2hval">${s.r2 >= 0 ? "+" : ""}${s.r2.toFixed(3)}</span>
    </div>`).join("");

  $("h2hCount").textContent = shared.length.toLocaleString();
  $("h2hVerdict").innerHTML =
    `The forest is ahead by ${(f.r - n.r).toFixed(3)} in <i>r</i> and
     ${(n.mae - f.mae).toFixed(3)} pK in mean error.`;
}

// --- selected complex --------------------------------------------------------

function select(id) {
  const rec = byId[id];
  if (!rec || !rec.forest) return;
  const fr = rec.forest, nr = rec.network;

  $("selId").textContent = id.toUpperCase();
  $("selMeasured").textContent = fr.measured.toFixed(2);
  $("selForest").textContent = fr.pred.toFixed(2);
  $("selForestErr").textContent = absErr(fr.pred, fr.measured).toFixed(2);
  $("selSpread").textContent = fr.spread.toFixed(2);

  const netRow = $("selNetworkRow");
  if (nr) {
    netRow.hidden = false;
    $("selNetwork").textContent = nr.pred.toFixed(2);
    $("selNetworkErr").textContent = absErr(nr.pred, nr.measured).toFixed(2);
  } else {
    netRow.hidden = true;
  }

  const closer = nr && absErr(fr.pred, fr.measured) <= absErr(nr.pred, nr.measured);
  $("selVerdict").className = `verdict ${closer ? "ok" : "miss"}`;
  $("selVerdict").textContent = !nr
    ? `Off by ${absErr(fr.pred, fr.measured).toFixed(2)} pK.`
    : closer
      ? `The forest is closer here, by ${(absErr(nr.pred, nr.measured) - absErr(fr.pred, fr.measured)).toFixed(2)} pK.`
      : `The network is closer here, by ${(absErr(fr.pred, fr.measured) - absErr(nr.pred, nr.measured)).toFixed(2)} pK. It happens on about a third of complexes.`;

  $("parity").querySelectorAll("circle").forEach((c) =>
    c.classList.toggle("sel", c.dataset.id === id));

  const s = N.structures[id];
  $("noStructure").hidden = !!s;
  $("viewerWrap").hidden = !s;
  if (s) showStructure(s);
}

function showStructure(s) {
  if (!viewer) viewer = $3Dmol.createViewer($("viewer"), { backgroundAlpha: 0 });
  viewer.clear();
  const lines = [String(s.elements.length), "pocket+ligand"];
  s.elements.forEach((el, i) => {
    lines.push(`${el} ${s.xyz[i * 3]} ${s.xyz[i * 3 + 1]} ${s.xyz[i * 3 + 2]}`);
  });
  viewer.addModel(lines.join("\n"), "xyz");
  const ligand = { index: Array.from({ length: s.n_ligand }, (_, i) => i) };
  // The pocket was drawn with `line`, which renders bonds only and at one pixel
  // wide: pocket atoms whose neighbours fell outside the 6 A cut have no bond to
  // draw and disappeared entirely, and the rest were hairlines invisible against
  // the panel. Thin sticks plus a small sphere per atom show every atom whether
  // or not it kept a bond. One flat mid-grey, readable on both themes, so the
  // caption's "grey = pocket" is true -- greyCarbon still colours N blue and O
  // red, which is what made the pocket look like stray coloured specks.
  viewer.setStyle({}, { stick: { radius: 0.07, color: POCKET_GREY },
                        sphere: { radius: 0.13, color: POCKET_GREY } });
  viewer.setStyle(ligand, { stick: { radius: 0.21, colorscheme: "cyanCarbon" },
                            sphere: { radius: 0.34, colorscheme: "cyanCarbon" } });
  // Frame the whole complex, not just the ligand, or the pocket sits off-screen.
  viewer.zoomTo();
  viewer.zoom(1.1);
  viewer.render();
}

// --- summary table -----------------------------------------------------------

function fillSummary() {
  // The repeated-split summary describes the full 300-tree forest. The bundle
  // ships a 100-tree copy so the page is not a 20 MB download, and the plot above
  // is that copy's predictions, so both figures are shown rather than quoting the
  // larger model's numbers over the smaller model's plot.
  const d = F.demo_metrics;
  if (d) {
    $("mDemoR").textContent = `${d.pearson >= 0 ? "+" : ""}${d.pearson.toFixed(3)}`;
    $("mDemoMae").textContent = d.mae.toFixed(3);
    $("mDemoTrees").textContent = d.n_trees;
  }
  const m = F.metrics;
  const fmt = (k) => `${m[k].mean >= 0 && k !== "mae" && k !== "rmse" ? "+" : ""}${m[k].mean.toFixed(3)} ± ${m[k].sd.toFixed(3)}`;
  $("mR").textContent = fmt("pearson");
  $("mMae").textContent = fmt("mae");
  $("mR2").textContent = fmt("r2");
  $("mBaseline").textContent = m.baseline_mae.mean.toFixed(3);
  $("mRepeats").textContent = m.n_repeats;
  $("mTest").textContent = F.complexes.length.toLocaleString();
  $("mFeatures").textContent = F.feature_names.length;
  $("mStructs").textContent = Object.keys(N.structures).length;
  $("mDataset").textContent = F.dataset;
}

// --- wiring ------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", async () => {
  try {
    [F, N] = await Promise.all([
      fetch("affinity.json").then((r) => r.json()),
      fetch("fusion.json").then((r) => r.json()),
    ]);
  } catch (e) {
    $("error").textContent = `Could not load the predictions: ${e}`;
    $("error").hidden = false;
    return;
  }

  F.complexes.forEach((c) => { byId[c.id] = { ...(byId[c.id] || {}), forest: c }; });
  N.complexes.forEach((c) => { byId[c.id] = { ...(byId[c.id] || {}), network: c }; });

  fillSummary();
  drawParity();
  drawHeadToHead();
  drawForestUncertainty();
  wireUpload();
  $("loading").hidden = true;
  $("app").hidden = false;

  const withStructure = Object.keys(N.structures).filter((id) => byId[id]?.forest);
  const pool = () => withStructure.map((id) => byId[id].forest);
  $("best").addEventListener("click", () =>
    select(pool().sort((a, b) => absErr(a.pred, a.measured) - absErr(b.pred, b.measured))[0].id));
  $("worst").addEventListener("click", () =>
    select(pool().sort((a, b) => absErr(b.pred, b.measured) - absErr(a.pred, a.measured))[0].id));
  $("disagree").addEventListener("click", () => {
    const both = withStructure.filter((id) => byId[id].network);
    both.sort((a, b) =>
      Math.abs(byId[b].forest.pred - byId[b].network.pred) -
      Math.abs(byId[a].forest.pred - byId[a].network.pred));
    select(both[0]);
  });

  select(withStructure[0]);
});

// --- the network's uncertainty, which the forest does not produce -------------
// Kept because it is a result in its own right: the graph model reports an
// epistemic and an aleatoric term, and neither predicts which answers are wrong.

function spearman(xs, ys) {
  const rank = (a) => {
    const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(a.length);
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      i = j + 1;
    }
    return r;
  };
  const rx = rank(xs), ry = rank(ys), n = xs.length;
  const mx = rx.reduce((s, v) => s + v, 0) / n;
  const my = ry.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = rx[i] - mx, b = ry[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return num / Math.sqrt(dx * dy);
}


function drawForestUncertainty() {
  // The forest is the shipped model, so this panel describes the forest. The
  // tree spread rides in the forest bundle; the Tanimoto distance to training
  // chemistry is only computed in the network's build, so it is joined across
  // the two by pdb id. Both bundles are exported from the same scaffold split
  // (seed 0), which is what makes that join legitimate -- when they drifted
  // apart earlier they overlapped on 458 of 740 complexes and every comparison
  // drawn from them was meaningless. Complexes missing from either side are
  // dropped rather than guessed at.
  const rows = F.complexes
    .map((f) => ({ f, n: byId[f.id] && byId[f.id].network }))
    .filter(({ f, n }) => n && n.ad_sim !== null && n.ad_sim !== undefined
                            && typeof f.spread === "number");
  if (!rows.length) { $("uncertaintyPanel").hidden = true; return; }

  const err = rows.map(({ f }) => absErr(f.pred, f.measured));
  const spread = rows.map(({ f }) => f.spread);
  const adDist = rows.map(({ n }) => 1 - n.ad_sim);

  const signals = [
    ["How much the 300 trees disagree", spearman(spread, err)],
    ["Distance to the nearest training ligand", spearman(adDist, err)],
  ];
  const best = Math.max(...signals.map(([, v]) => Math.abs(v)));
  $("signals").innerHTML = signals.map(([label, rho]) => `
    <div class="sigrow">
      <span class="siglabel">${label}</span>
      <span class="sigbar"><i class="${rho < 0 ? "neg" : ""}" style="width:${((Math.abs(rho) / best) * 100).toFixed(0)}%"></i></span>
      <span class="sigval">${rho >= 0 ? "+" : ""}${rho.toFixed(2)}</span>
    </div>`).join("");

  const t = N.ad ? N.ad.threshold : 0;
  const mae = (a) => a.reduce((s, { f }) => s + absErr(f.pred, f.measured), 0)
                     / Math.max(a.length, 1);
  const inside = rows.filter(({ n }) => n.ad_sim >= t);
  const outside = rows.filter(({ n }) => n.ad_sim < t);
  const cov1 = rows.filter(({ f }) => absErr(f.pred, f.measured) <= f.spread).length / rows.length;
  const cov2 = rows.filter(({ f }) => absErr(f.pred, f.measured) <= 2 * f.spread).length / rows.length;

  $("uncertaintyVerdict").innerHTML =
    `Treating the spread as a standard deviation,
     ${(cov1 * 100).toFixed(0)}% of measurements land inside ±1 spread and
     ${(cov2 * 100).toFixed(0)}% inside ±2, against 68% and 95% for a normal
     distribution. Close, and nothing was fitted to make it so — over five
     scaffold splits it runs a little wide, at 73% and 97%.
     <br><br>
     The ${outside.length} complexes with no close training analogue are wrong by
     ${mae(outside).toFixed(2)} pK on average against ${mae(inside).toFixed(2)} pK for
     the other ${inside.length}. Worth knowing, and the weaker of the two warnings.`;
}

// --- scoring an uploaded pose ------------------------------------------------
// The forest can run here, so it does. Features are built by docs/dock.js, whose
// agreement with the Python implementation is checked numerically by
// scripts/check_browser_features.py rather than assumed.

let dockForest = null;
let RDKitDock = null;
const uploaded = { receptor: null, ligand: null };

function solubilityWords(pk) {
  if (pk >= 9) return "Very tight binding, in the nanomolar range or below.";
  if (pk >= 7) return "Tight binding, roughly 10–100 nanomolar.";
  if (pk >= 5) return "Moderate binding, roughly micromolar.";
  return "Weak binding, in the millimolar range.";
}

/** How many ECFP4 bits this bundle's forest was trained with; 0 if none. */
const fpBits = () => (F && F.fingerprint_bits) || 0;

/** Descriptors, heavy-atom count and fingerprint for a SMILES, or null.
 *
 *  Delegates to dock.js so the page and scripts/check_browser_features.py run
 *  the same code. That helper also strips salts, which this page previously did
 *  not while Python always did -- a multi-fragment SMILES was featurised whole
 *  here and as its largest fragment there. */
function ligandFromSmiles(smiles) {
  if (!RDKitDock) return null;
  return window.CADDackDock.ligandFeatures(RDKitDock, smiles, fpBits());
}

function scorePose() {
  const err = $("dockError");
  const show = (msg) => { err.textContent = msg; err.hidden = false;
                          $("dockResult").hidden = true; };

  if (!uploaded.receptor || !uploaded.ligand) {
    return show("Choose both a receptor and a ligand PDBQT.");
  }
  const D = window.CADDackDock;
  const rec = D.parsePdbqt(uploaded.receptor);
  const lig = D.parsePdbqt(uploaded.ligand);
  if (!lig.z.length) return show("No atoms found in the ligand file. Is it a PDBQT?");
  if (!rec.z.length) return show("No atoms found in the receptor file. Is it a PDBQT?");

  const smiles = $("dockSmiles").value.trim();
  let desc = smiles ? ligandFromSmiles(smiles) : null;
  if (smiles && !desc) return show(`Could not parse that SMILES: ${smiles}`);

  const approximate = !desc;
  if (approximate) {
    // Without a SMILES the descriptors are unavailable, and so is the
    // fingerprint. Zeroing six of 113 columns left the interaction features
    // carrying the prediction; zeroing 1030 of 1137 does not, so when the model
    // uses a fingerprint the SMILES is required rather than guessed at.
    if (fpBits()) {
      return show("This model reads the ligand's structure as well as its pose, "
                + "so it needs the ligand SMILES. Paste it above.");
    }
    desc = { descriptors: { MolWt: 0, LogP: 0, TPSA: 0, NumHBD: 0, NumHBA: 0,
                            NumRotBonds: 0 }, heavy: lig.z.filter((z) => z !== 1).length,
             fp: null };
  }

  const z = [...lig.z, ...rec.z];
  const xyz = [...lig.xyz, ...rec.xyz];
  const inter = D.interactionFeatures(z, xyz, lig.z.length);
  if (inter.nContacts === 0) {
    return show("The ligand makes no contacts within 8 Å of the receptor. "
              + "Are the two files in the same coordinate frame?");
  }

  const x = D.buildFeatureVector(desc.descriptors, desc.heavy, inter, desc.fp);
  if (x.length !== F.feature_names.length) {
    return show(`Built ${x.length} features but the model expects ${F.feature_names.length}.`);
  }

  const perTree = D.predictPerTree(dockForest, x);
  const mean = perTree.reduce((s, v) => s + v, 0) / perTree.length;
  const sd = Math.sqrt(perTree.reduce((s, v) => s + (v - mean) ** 2, 0) / perTree.length);

  err.hidden = true;
  $("dockPred").textContent = mean.toFixed(2);
  $("dockSpread").textContent = sd.toFixed(2);
  $("dockPlain").textContent = solubilityWords(mean);
  $("dockNLig").textContent = lig.z.length.toLocaleString();
  $("dockNRec").textContent = rec.z.length.toLocaleString();
  $("dockNContacts").textContent = inter.nContacts.toLocaleString();
  $("dockVina").textContent = inter.vina_inter.toFixed(2);

  const notes = [];
  if (approximate) {
    notes.push("No SMILES was given, so the six ligand descriptors are zero rather "
             + "than measured. The interaction features are still exact, but treat "
             + "the number as indicative.");
  }
  notes.push("The model was trained on crystal structures. A docked pose is a "
           + "prediction, so an error in the pose becomes an error here, and this "
           + "says how good <em>this</em> pose looks rather than how well the molecule "
           + "binds.");
  $("dockCaveat").innerHTML = notes.join(" ");
  $("dockResult").hidden = false;
}

function wireUpload() {
  dockForest = window.CADDackDock.decodeForest(F.forest);

  const bind = (inputId, nameId, key) => {
    $(inputId).addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      $(nameId).textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
      const reader = new FileReader();
      reader.onload = () => { uploaded[key] = reader.result; };
      reader.onerror = () => {
        $("dockError").textContent = `Could not read ${file.name}.`;
        $("dockError").hidden = false;
      };
      reader.readAsText(file);
    });
  };
  bind("recFile", "recName", "receptor");
  bind("ligFile", "ligName", "ligand");
  $("dockRun").addEventListener("click", scorePose);

  // RDKit is only needed for the descriptors, so the panel works without waiting.
  if (typeof initRDKitModule === "function") {
    initRDKitModule().then((rdkit) => { RDKitDock = rdkit; }).catch(() => {});
  }
}
