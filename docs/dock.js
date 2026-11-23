/* ---------------------------------------------------------------------------
   Score a docked pose in the browser.

   The shipped affinity model is a random forest, so it can be evaluated here the
   way the similarity page evaluates its own: walk the trees. The graph network it
   replaced could not be — its topology is built inside the forward pass — which is
   part of why the forest is what ships.

   113 features are needed. 107 of them (Vina's interaction terms and the binned
   contact counts) come from coordinates alone and are ported from
   caddack.docking.scoring. The remaining six descriptors plus the ligand's heavy
   atom count need the molecule's bonding, which a PDBQT does not record, so a
   SMILES is asked for; without one, bonds are inferred from the coordinates and
   the descriptors are approximate.

   Every constant here mirrors the Python side. tests/test_scoring.py pins the
   Python definitions, and scripts/check_browser_features.mjs compares the two
   numerically on real complexes.
   --------------------------------------------------------------------------- */

// --- PDBQT ------------------------------------------------------------------

// AutoDock atom types carry hydrogen-bonding and aromaticity information in a
// suffix: OA is an acceptor oxygen, HD a polar hydrogen, A an aromatic carbon.
// Only the element matters for the features, so the suffix is dropped.
const AD_TYPE_TO_ELEMENT = {
  A: "C", C: "C", N: "N", NA: "N", NS: "N", OA: "O", OS: "O", O: "O",
  SA: "S", S: "S", P: "P", F: "F", Cl: "Cl", CL: "Cl", Br: "Br", BR: "Br",
  I: "I", H: "H", HD: "H", HS: "H",
  Mg: "Mg", MG: "Mg", Zn: "Zn", ZN: "Zn", Fe: "Fe", FE: "Fe",
  Ca: "Ca", CA: "Ca", Mn: "Mn", MN: "Mn", Cu: "Cu", CU: "Cu",
  Na: "Na", K: "K", Se: "Se", SE: "Se",
};
const ELEMENT_Z = { H: 1, B: 5, C: 6, N: 7, O: 8, F: 9, Na: 11, Mg: 12, Si: 14,
  P: 15, S: 16, Cl: 17, K: 19, Ca: 20, Mn: 25, Fe: 26, Cu: 29, Zn: 30,
  Se: 34, Br: 35, I: 53 };

/** Parse ATOM/HETATM records out of a PDBQT. Returns atomic numbers and coords.
 *
 *  Only the first MODEL is read: a Vina output holds one block per pose, and
 *  scoring the concatenation of all of them would be meaningless.
 */
function parsePdbqt(text) {
  const z = [], xyz = [];
  let seenModel = false;
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6).trim();
    if (record === "MODEL") {
      if (seenModel) break;          // second pose: stop
      seenModel = true;
      continue;
    }
    if (record === "ENDMDL") break;
    if (record !== "ATOM" && record !== "HETATM") continue;

    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const zc = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(zc)) continue;

    // Columns 78-79 hold the AutoDock type. Fall back to the atom name when a
    // file is non-conforming, which hand-edited PDBQTs often are.
    let type = line.slice(77, 79).trim();
    if (!type) type = line.slice(12, 16).trim().replace(/[0-9]/g, "").slice(0, 2);
    const element = AD_TYPE_TO_ELEMENT[type]
      || AD_TYPE_TO_ELEMENT[type.toUpperCase()]
      || (ELEMENT_Z[type[0]] ? type[0] : null);
    if (!element) continue;

    z.push(ELEMENT_Z[element] ?? 6);
    xyz.push([x, y, zc]);
  }
  return { z, xyz };
}

// --- interaction features (ported from caddack.docking.scoring) ---------------

const VDW_RADII = { 6: 1.9, 7: 1.8, 8: 1.7, 15: 2.1, 16: 2.0,
                    9: 1.5, 17: 1.8, 35: 2.0, 53: 2.2 };
const METAL_RADIUS = 1.2;
const HALOGENS = new Set([9, 17, 35, 53]);
const POLAR = new Set([7, 8]);
const PAIR_CUTOFF = 8.0;
const BOND_CUTOFF = 1.95;
const VINA_WEIGHTS = { gauss1: -0.035579, gauss2: -0.005156, repulsion: 0.840245,
                       hydrophobic: -0.035069, hbond: -0.587439 };
const CONTACT_CLASSES = [[6], [7], [8], [16, 15, 9, 17, 35, 53]];
const N_CLASSES = CONTACT_CLASSES.length + 1;
const CONTACT_SHELLS = [3.5, 4.5, 6.0, 8.0];

const radius = (zi) => VDW_RADII[zi] ?? METAL_RADIUS;
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function classIndex(zi) {
  for (let i = 0; i < CONTACT_CLASSES.length; i++) {
    if (CONTACT_CLASSES[i].includes(zi)) return i;
  }
  return N_CLASSES - 1;
}

/** Carbon not bonded to N or O, plus halogens. A polarised carbon is excluded,
 *  which is why bonded neighbours have to be resolved rather than read off. */
function hydrophobicMask(z, xyz) {
  return z.map((zi, i) => {
    if (HALOGENS.has(zi)) return true;
    if (zi !== 6) return false;
    for (let j = 0; j < z.length; j++) {
      if (j !== i && POLAR.has(z[j]) && dist(xyz[i], xyz[j]) < BOND_CUTOFF) return false;
    }
    return true;
  });
}

/** Vina's five terms plus their weighted sum, over ligand-pocket pairs only.
 *  Hydrogens are dropped first: Vina is united-atom, and H has no radius here. */
function interactionFeatures(zAll, xyzAll, nLigandAll) {
  const keep = zAll.map((zi) => zi !== 1);
  const z = [], xyz = [];
  let nLigand = 0;
  zAll.forEach((zi, i) => {
    if (!keep[i]) return;
    if (i < nLigandAll) nLigand++;
    z.push(zi); xyz.push(xyzAll[i]);
  });

  const terms = { gauss1: 0, gauss2: 0, repulsion: 0, hydrophobic: 0, hbond: 0 };
  const contacts = new Float64Array(N_CLASSES * N_CLASSES * CONTACT_SHELLS.length);
  let nContacts = 0;
  if (nLigand === 0 || nLigand >= z.length) {
    return { terms, vina_inter: 0, contacts, nContacts: 0, nLigandHeavy: nLigand };
  }

  const hydro = hydrophobicMask(z, xyz);
  const perShell = N_CLASSES * N_CLASSES;

  for (let i = 0; i < nLigand; i++) {
    for (let j = nLigand; j < z.length; j++) {
      const r = dist(xyz[i], xyz[j]);
      if (r > PAIR_CUTOFF) continue;
      nContacts++;

      const d = r - (radius(z[i]) + radius(z[j]));
      terms.gauss1 += Math.exp(-((d / 0.5) ** 2));
      terms.gauss2 += Math.exp(-(((d - 3.0) / 2.0) ** 2));
      if (d < 0) terms.repulsion += d * d;
      if (hydro[i] && hydro[j]) {
        terms.hydrophobic += Math.min(1, Math.max(0, 1.5 - d));
      }
      if (POLAR.has(z[i]) && POLAR.has(z[j])) {
        terms.hbond += Math.min(1, Math.max(0, -d / 0.7));
      }

      // Contact bin: element-class pair, then distance shell.
      let shell = -1;
      let lower = 0;
      for (let s = 0; s < CONTACT_SHELLS.length; s++) {
        if (r > lower && r <= CONTACT_SHELLS[s]) { shell = s; break; }
        lower = CONTACT_SHELLS[s];
      }
      if (shell >= 0) {
        contacts[shell * perShell + classIndex(z[i]) * N_CLASSES + classIndex(z[j])] += 1;
      }
    }
  }

  for (let k = 0; k < contacts.length; k++) contacts[k] /= Math.max(nLigand, 1);
  const vina_inter = Object.keys(VINA_WEIGHTS)
    .reduce((s, k) => s + VINA_WEIGHTS[k] * terms[k], 0);
  return { terms, vina_inter, contacts, nContacts, nLigandHeavy: nLigand };
}

/** Assemble the full feature vector in the order the model expects. */
function buildFeatureVector(descriptors, nLigandHeavy, inter) {
  return Float64Array.from([
    descriptors.MolWt, descriptors.LogP, descriptors.TPSA,
    descriptors.NumHBD, descriptors.NumHBA, descriptors.NumRotBonds,
    nLigandHeavy,
    inter.terms.gauss1, inter.terms.gauss2, inter.terms.repulsion,
    inter.terms.hydrophobic, inter.terms.hbond, inter.vina_inter,
    ...inter.contacts,
  ]);
}

// --- forest ------------------------------------------------------------------

function decodeForest(f) {
  const bytes = (b64) => {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  };
  return {
    nTrees: f.n_trees, offsets: f.offsets,
    feature: new Int16Array(bytes(f.feature)),
    threshold: new Float32Array(bytes(f.threshold)),
    left: new Int32Array(bytes(f.left)),
    right: new Int32Array(bytes(f.right)),
    value: new Float32Array(bytes(f.value)),
  };
}

function predictPerTree(forest, x) {
  const { nTrees, offsets, feature, threshold, left, right, value } = forest;
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

window.CADDackDock = {
  parsePdbqt, interactionFeatures, buildFeatureVector, decodeForest, predictPerTree,
  ELEMENT_Z,
};
