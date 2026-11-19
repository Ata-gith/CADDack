/* ---------------------------------------------------------------------------
   Fusion-model explorer.

   Unlike the solubility page, this one does not run the model. FusionAffinityNet
   needs a protein-ligand complex with 3D coordinates, which a visitor cannot type
   in, and its Bayesian head needs Monte Carlo sampling over weight posteriors.
   So docs/fusion.json carries predictions computed offline by
   scripts/build_fusion_demo.py, and the page is an explorer over them.

   What that still buys: you can check whether the uncertainty the model reports
   actually tracks the errors it makes. That is the claim worth testing, and a
   static table cannot show it.
   --------------------------------------------------------------------------- */

const DATA_URL = "fusion.json";

let D = null;          // the loaded bundle
let viewer = null;     // 3Dmol viewer, created once
let selected = null;

const $ = (id) => document.getElementById(id);
const err = (r) => Math.abs(r.pred - r.measured);
const totalSd = (r) => Math.hypot(r.epi, r.ale);

// --- parity plot -------------------------------------------------------------

const PLOT = { w: 520, h: 520, pad: 48, lo: 1, hi: 13 };

function plotXY(v, axis) {
  const { pad, w, h, lo, hi } = PLOT;
  const t = (v - lo) / (hi - lo);
  return axis === "x" ? pad + t * (w - 2 * pad) : h - pad - t * (h - 2 * pad);
}

/** Predicted against measured, with uncertainty as the visual channel. */
function drawParity() {
  const { w, h, pad, lo, hi } = PLOT;
  const ticks = [2, 4, 6, 8, 10, 12];
  const withStructure = new Set(Object.keys(D.structures));

  const grid = ticks.map((t) => `
    <line x1="${plotXY(t, "x")}" y1="${pad}" x2="${plotXY(t, "x")}" y2="${h - pad}" class="grid"/>
    <line x1="${pad}" y1="${plotXY(t, "y")}" x2="${w - pad}" y2="${plotXY(t, "y")}" class="grid"/>
    <text x="${plotXY(t, "x")}" y="${h - pad + 16}" class="tick" text-anchor="middle">${t}</text>
    <text x="${pad - 8}" y="${plotXY(t, "y") + 4}" class="tick" text-anchor="end">${t}</text>`).join("");

  // Points are sorted so the interesting ones (shippable structures) draw last.
  const pts = [...D.complexes]
    .sort((a, b) => withStructure.has(a.id) - withStructure.has(b.id))
    .map((r) => {
      const sd = totalSd(r);
      const hasStruct = withStructure.has(r.id);
      return `<circle cx="${plotXY(r.measured, "x").toFixed(1)}"
                      cy="${plotXY(r.pred, "y").toFixed(1)}"
                      r="${(2.2 + sd * 1.6).toFixed(1)}"
                      class="pt${hasStruct ? " has3d" : ""}"
                      data-id="${r.id}"><title>${r.id}: measured ${r.measured}, predicted ${r.pred}</title></circle>`;
    }).join("");

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

// --- does the uncertainty mean anything? ------------------------------------

/** Spearman rank correlation: does claimed uncertainty rank actual error? */
function spearman(xs, ys) {
  const rank = (a) => {
    const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(a.length);
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2 + 1;          // average rank for ties
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

function drawReliability() {
  const R = D.reliability;
  const w = 460, h = 240, pad = 46;
  const maxV = Math.max(...R.map((b) => Math.max(b.mean_abs_error, b.mean_uncertainty))) * 1.15;
  const bw = (w - 2 * pad) / R.length;
  const y = (v) => h - pad - (v / maxV) * (h - 2 * pad);

  const bars = R.map((b, i) => {
    const x = pad + i * bw;
    return `
      <rect x="${x + bw * 0.12}" y="${y(b.mean_abs_error)}" width="${bw * 0.34}"
            height="${h - pad - y(b.mean_abs_error)}" class="bar err"/>
      <rect x="${x + bw * 0.52}" y="${y(b.mean_uncertainty)}" width="${bw * 0.34}"
            height="${h - pad - y(b.mean_uncertainty)}" class="bar unc"/>
      <text x="${x + bw / 2}" y="${h - pad + 15}" class="tick" text-anchor="middle">Q${i + 1}</text>`;
  }).join("");

  $("reliability").innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Error by uncertainty quintile">
      <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" class="grid"/>
      ${bars}
      <text x="${w / 2}" y="${h - 6}" class="axis" text-anchor="middle">
        test complexes grouped by how uncertain the model said it was →</text>
    </svg>`;

  // Two different questions, and here they have different answers.
  const rho = spearman(D.complexes.map(totalSd), D.complexes.map(err));
  const ratio = R[R.length - 1].mean_abs_error / R[0].mean_abs_error;
  const m = D.metrics;

  // The quintile ratio compares only the end bins. If the bins in between do not
  // rise in order, that ratio flatters a trend the chart plainly does not show,
  // so say which it is rather than quoting the convenient number.
  const errs = R.map((b) => b.mean_abs_error);
  const monotonic = errs.every((v, i) => i === 0 || v >= errs[i - 1]);

  $("reliabilityVerdict").innerHTML =
    `<strong>Partly, and it's worth separating two claims.</strong> The intervals are
     well calibrated <em>in aggregate</em>: ${(m.coverage_1sigma * 100).toFixed(0)}% of
     measurements fall inside 1σ and ${(m.coverage_2sigma * 100).toFixed(0)}% inside 2σ,
     against an ideal of 68% and 95%. So if you ask "how often is the truth inside the
     error bars?", the answer is honest.
     <br><br>
     Telling <em>which individual</em> predictions are bad is a harder job, and here it
     barely manages. Error rises ${ratio.toFixed(2)}× from the most confident fifth of
     the test set to the least${monotonic ? "" : ", and not even steadily — the middle"
     + " bins do not fall in order, so that ratio flatters the trend"}. Across all
     ${D.complexes.length} complexes the rank correlation between claimed uncertainty and
     actual error is ρ&nbsp;=&nbsp;${rho.toFixed(2)}, which is close to useless as a
     per-prediction warning.
     <br><br>
     A well-calibrated average is not the same thing as knowing which answers to doubt,
     and conflating the two is how uncertainty estimates get oversold. This is also one
     split: treat ρ as indicative, not measured.`;
}

// --- selected complex --------------------------------------------------------

function select(id) {
  const r = D.complexes.find((c) => c.id === id);
  if (!r) return;
  selected = r;

  $("selId").textContent = r.id.toUpperCase();
  $("selPred").textContent = r.pred.toFixed(2);
  $("selMeasured").textContent = r.measured.toFixed(2);
  $("selError").textContent = err(r).toFixed(2);
  $("selEpi").textContent = r.epi.toFixed(2);
  $("selAle").textContent = r.ale.toFixed(2);
  $("selAtoms").textContent = `${r.n_ligand} ligand · ${r.n_pocket} pocket`;

  const sd = totalSd(r);
  const within = err(r) <= 2 * sd;
  $("selVerdict").className = `verdict ${within ? "ok" : "miss"}`;
  $("selVerdict").textContent = within
    ? `Off by ${err(r).toFixed(2)} pK, inside its own 2σ interval of ±${(2 * sd).toFixed(2)}.`
    : `Off by ${err(r).toFixed(2)} pK, outside its own 2σ interval of ±${(2 * sd).toFixed(2)} — `
      + `a confident miss, which is the failure that matters.`;

  $("parity").querySelectorAll("circle").forEach((c) =>
    c.classList.toggle("sel", c.dataset.id === id));

  const s = D.structures[r.id];
  $("noStructure").hidden = !!s;
  $("viewerWrap").hidden = !s;
  if (s) showStructure(s);
}

/** Render the pocket and ligand. Coordinates ship as a flat array. */
function showStructure(s) {
  if (!viewer) {
    viewer = $3Dmol.createViewer($("viewer"), { backgroundAlpha: 0 });
  }
  viewer.clear();

  // 3Dmol reads XYZ directly, which avoids inventing PDB records we do not have.
  const lines = [String(s.elements.length), "pocket+ligand"];
  s.elements.forEach((el, i) => {
    lines.push(`${el} ${s.xyz[i * 3]} ${s.xyz[i * 3 + 1]} ${s.xyz[i * 3 + 2]}`);
  });
  viewer.addModel(lines.join("\n"), "xyz");

  // Ligand atoms come first in the export; everything after is pocket.
  const ligand = { index: Array.from({ length: s.n_ligand }, (_, i) => i) };
  viewer.setStyle({}, { line: { colorscheme: "greyCarbon", opacity: 0.55 } });
  viewer.setStyle(ligand, { stick: { radius: 0.21, colorscheme: "cyanCarbon" },
                            sphere: { radius: 0.34, colorscheme: "cyanCarbon" } });
  viewer.zoomTo(ligand);
  viewer.zoom(0.75);
  viewer.render();
}

// --- wiring ------------------------------------------------------------------

function fillSummary() {
  const m = D.metrics;
  $("mMae").textContent = m.mae.toFixed(2);
  $("mR2").textContent = m.r2.toFixed(2);
  $("mR").textContent = m.pearson.toFixed(2);
  $("mCov1").textContent = `${(m.coverage_1sigma * 100).toFixed(0)}%`;
  $("mCov2").textContent = `${(m.coverage_2sigma * 100).toFixed(0)}%`;
  $("mBaseline").textContent = m.baseline_mae.toFixed(2);
  $("mTrain").textContent = m.n_train.toLocaleString();
  $("mTest").textContent = m.n_test.toLocaleString();
  $("mGroups").textContent = D.split.n_scaffold_groups.toLocaleString();
  $("mStructs").textContent = Object.keys(D.structures).length;
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    D = await (await fetch(DATA_URL)).json();
  } catch (e) {
    $("error").textContent = `Could not load the predictions: ${e}`;
    $("error").hidden = false;
    return;
  }
  fillSummary();
  drawParity();
  drawReliability();
  $("loading").hidden = true;
  $("app").hidden = false;

  const withStructure = Object.keys(D.structures);
  $("worst").addEventListener("click", () => {
    const pool = D.complexes.filter((r) => withStructure.includes(r.id));
    select(pool.sort((a, b) => err(b) - err(a))[0].id);
  });
  $("confidentMiss").addEventListener("click", () => {
    const pool = D.complexes.filter((r) => withStructure.includes(r.id));
    select(pool.sort((a, b) => (err(b) / totalSd(b)) - (err(a) / totalSd(a)))[0].id);
  });
  $("best").addEventListener("click", () => {
    const pool = D.complexes.filter((r) => withStructure.includes(r.id));
    select(pool.sort((a, b) => err(a) - err(b))[0].id);
  });

  // Open on a confident miss: the page asks whether the model knows when it is
  // wrong, so the first thing a visitor sees should be a case where it did not.
  const pool = D.complexes.filter((r) => withStructure.includes(r.id));
  const opener = pool.sort((a, b) => (err(b) / totalSd(b)) - (err(a) / totalSd(a)))[0];
  select((opener ?? D.complexes[0]).id);
});
