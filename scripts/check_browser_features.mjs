#!/usr/bin/env node
/* Compare the browser's feature computation against Python's, on real complexes.
 *
 * docs/dock.js reimplements caddack.docking.scoring in JavaScript so a docked
 * pose can be scored in the browser. A silent divergence there would produce
 * confident, wrong affinities with nothing to indicate it — the same class of
 * failure that the similarity page's chirality flag caused — so the two are
 * compared numerically rather than assumed to agree.
 *
 * Driven by scripts/check_browser_features.py, which writes the PDBQT fixtures
 * and the expected values, then calls this to produce the browser's.
 *
 *   node scripts/check_browser_features.mjs <fixtures.json>
 */
import pkg from '/opt/node-tools/node_modules/playwright/index.js';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = pkg;
const fixtures = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const dockJs = readFileSync(new URL('../docs/dock.js', import.meta.url), 'utf8');
const docsDir = fileURLToPath(new URL('../docs/', import.meta.url));

// RDKit's loader fetches its .wasm over HTTP relative to the page, so the page
// needs a real origin: a file:// or about:blank page cannot fetch it. Serve
// docs/ for the life of the check.
const types = { '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const file = join(docsDir, req.url.split('?')[0].replace(/^\/+/, ''));
  if (!file.startsWith(docsDir) || !existsSync(file) || statSync(file).isDirectory()) {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    return res.end('<!doctype html><title>check</title>');
  }
  res.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`${origin}/`);
await page.setContent('<!doctype html><title>check</title>');
await page.addScriptTag({ content: dockJs });

// The fingerprint half of the vector needs RDKit, exactly as the page does.
if (fixtures.fp_bits > 0) {
  await page.addScriptTag({ url: `${origin}/RDKit_minimal.js` });
  await page.evaluate(async () => { window.RDKit = await window.initRDKitModule(); });
}

const results = [];
for (const fx of fixtures.cases) {
  const out = await page.evaluate(({ receptor, ligand, descriptors, smiles, fpBits }) => {
    const D = window.CADDackDock;
    const lig = D.parsePdbqt(ligand);
    const rec = D.parsePdbqt(receptor);
    const z = [...lig.z, ...rec.z];
    const xyz = [...lig.xyz, ...rec.xyz];
    const inter = D.interactionFeatures(z, xyz, lig.z.length);
    // When the model carries a fingerprint, take the whole ligand block from
    // the same helper the page uses, so the SMILES path is checked too and not
    // just the geometry.
    const fromSmiles = fpBits ? D.ligandFeatures(window.RDKit, smiles, fpBits) : null;
    const x = D.buildFeatureVector(descriptors, inter.nLigandHeavy, inter,
                                   fromSmiles && fromSmiles.fp);
    return {
      n_ligand_parsed: lig.z.length,
      n_receptor_parsed: rec.z.length,
      features: Array.from(x),
      browser_descriptors: fromSmiles ? fromSmiles.descriptors : null,
      browser_heavy: fromSmiles ? fromSmiles.heavy : null,
    };
  }, { ...fx, fpBits: fixtures.fp_bits });
  results.push({ id: fx.id, ...out });
}

console.log(JSON.stringify({ results }));
await browser.close();
server.close();
