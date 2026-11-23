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
import { readFileSync } from 'node:fs';

const { chromium } = pkg;
const fixtures = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const dockJs = readFileSync(new URL('../docs/dock.js', import.meta.url), 'utf8');

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent('<!doctype html><title>check</title>');
await page.addScriptTag({ content: dockJs });

const results = [];
for (const fx of fixtures.cases) {
  const out = await page.evaluate(({ receptor, ligand, descriptors }) => {
    const D = window.CADDackDock;
    const lig = D.parsePdbqt(ligand);
    const rec = D.parsePdbqt(receptor);
    const z = [...lig.z, ...rec.z];
    const xyz = [...lig.xyz, ...rec.xyz];
    const inter = D.interactionFeatures(z, xyz, lig.z.length);
    const x = D.buildFeatureVector(descriptors, inter.nLigandHeavy, inter);
    return {
      n_ligand_parsed: lig.z.length,
      n_receptor_parsed: rec.z.length,
      features: Array.from(x),
    };
  }, fx);
  results.push({ id: fx.id, ...out });
}

console.log(JSON.stringify({ results }));
await browser.close();
