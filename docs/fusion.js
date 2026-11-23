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
    ["Random forest, 115 features", f, true],
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
    `On the same ${shared.length} held-out complexes the forest is ahead by
     ${(f.r - n.r).toFixed(3)} in <i>r</i> and ${(n.mae - f.mae).toFixed(3)} pK in
     mean error. Giving the network the same 115 features narrowed the gap but did
     not close it — r 0.626 against the forest's 0.696 on a matched split — so the
     two graph encoders are adding 420,000 parameters without adding accuracy.
     On 2,959 training complexes that is the ordinary outcome: bagged trees on
     well-built features are hard to beat at this scale.`;
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
  viewer.setStyle({}, { line: { colorscheme: "greyCarbon", opacity: 0.55 } });
  viewer.setStyle(ligand, { stick: { radius: 0.21, colorscheme: "cyanCarbon" },
                            sphere: { radius: 0.34, colorscheme: "cyanCarbon" } });
  viewer.zoomTo(ligand);
  viewer.zoom(0.75);
  viewer.render();
}

// --- summary table -----------------------------------------------------------

function fillSummary() {
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
  drawNetworkUncertainty();
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

function rankAverage(a, b) {
  const ranks = (xs) => {
    const idx = xs.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(xs.length);
    idx.forEach(([, i], k) => { r[i] = k; });
    return r;
  };
  const ra = ranks(a), rb = ranks(b);
  return ra.map((v, i) => v + rb[i]);
}

function drawNetworkUncertainty() {
  const rows = N.complexes.filter((r) => r.ad_sim !== null && r.ad_sim !== undefined);
  if (!rows.length) { $("uncertaintyPanel").hidden = true; return; }

  const err = rows.map((r) => absErr(r.pred, r.measured));
  const epi = rows.map((r) => r.epi);
  const ale = rows.map((r) => r.ale);
  const total = rows.map((r) => Math.hypot(r.epi, r.ale));
  const adDist = rows.map((r) => 1 - r.ad_sim);
  const combined = rankAverage(total, adDist);

  const signals = [
    ["epistemic σ — the model's own doubt", spearman(epi, err)],
    ["aleatoric σ — predicted noise", spearman(ale, err)],
    ["both, as the model reports them", spearman(total, err)],
    ["similarity to the training ligands", spearman(adDist, err)],
    ["the last two, rank-averaged", spearman(combined, err)],
  ];
  const best = Math.max(...signals.map(([, v]) => Math.abs(v)));

  $("signals").innerHTML = signals.map(([label, rho]) => `
    <div class="sigrow">
      <span class="siglabel">${label}</span>
      <span class="sigbar"><i class="${rho < 0 ? "neg" : ""}" style="width:${((Math.abs(rho) / best) * 100).toFixed(0)}%"></i></span>
      <span class="sigval">${rho >= 0 ? "+" : ""}${rho.toFixed(3)}</span>
    </div>`).join("");

  const m = N.metrics;
  const epiRho = spearman(epi, err);
  const adRho = spearman(adDist, err);
  const combRho = spearman(combined, err);
  const t = N.ad ? N.ad.threshold : 0;
  const outside = rows.filter((r) => r.ad_sim < t);
  const inside = rows.filter((r) => r.ad_sim >= t);
  const mae = (a) => a.reduce((s, r) => s + absErr(r.pred, r.measured), 0) / Math.max(a.length, 1);

  $("uncertaintyVerdict").innerHTML =
    `<strong>The widths are right; the ordering is not.</strong>
     ${(m.coverage_1sigma * 100).toFixed(0)}% of measurements land inside 1σ and
     ${(m.coverage_2sigma * 100).toFixed(0)}% inside 2σ, against an ideal of 68% and
     95%. But across all ${rows.length} complexes the rank correlation between the
     stated uncertainty and the error actually made is only
     ρ&nbsp;=&nbsp;${spearman(total, err).toFixed(2)}, and the epistemic half alone
     scores ${epiRho.toFixed(3)} — at or below zero.
     <br><br>
     A measure that ignores the model entirely does better: how similar the ligand
     is to the training set scores ρ&nbsp;=&nbsp;${adRho.toFixed(3)}, and the
     ${outside.length} complexes outside that similarity threshold average
     ${mae(outside).toFixed(2)} pK of error against ${mae(inside).toFixed(2)} for
     those inside. Combining the two beats either alone
     (${combRho.toFixed(3)}).
     <br><br>
     Three mechanisms were tried for this — a Bayesian posterior, dropout sampling,
     and an ensemble of five independently trained networks — and none beat the
     similarity lookup. Read the middle rows as tied: repeating the measurement
     moves each by about ±0.02.`;
}
