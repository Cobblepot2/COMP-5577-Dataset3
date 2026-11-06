/* Rebuild the report figure from archived measurements; requires sharp. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const sourcePath = path.join(root, 'docs/results/public85/study_summary.json');
const referencePath = path.join(root, 'docs/results/defense_pgd_augmented.json');
const outputPath = process.argv[2] || path.join(root, 'docs/assets/training_source_curves.png');
const study = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
const referenceReport = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
const baseline = Math.min(...referenceReport.attacks
  .filter(a => ['transfer_fgsm8', 'transfer_pgd20x3'].includes(a.settings.name))
  .map(a => a.mixed_50pct_metrics.macro_f1));
if (!Number.isFinite(baseline)) throw new Error('Missing archived reference measurements');

const width = 2250, height = 1021;
const left = 188, right = 2188, top = 237, bottom = 866;
const x = epoch => left + epoch / 40 * (right - left);
const y = value => bottom - value * (bottom - top);
const dash = '21 12.333';
const referenceStyle = `stroke="#63738a" stroke-width="4.5" stroke-dasharray="${dash}" stroke-linecap="round" fill="none"`;
const series = [
  {name: 'improve_trades_beta03', label: 'TRADES beta 0.3', color: '#c4590f', legendX: 75},
  {name: 'improve_trades_beta1', label: 'TRADES beta 1', color: '#1275a8', legendX: 630},
  {name: 'improve_hybrid', label: 'Hybrid PGD', color: '#087d66', legendX: 1115},
];
const records = series.map(item => {
  const trial = study.trials_40epochs.find(t => t.name === item.name);
  const values = trial.epochs.map(e => [e.epoch, e.criterion.worst_source_mixed_macro_f1]);
  if (values.length !== 40 || values.some(([epoch, value], i) => epoch !== i + 1 || !(value >= 0 && value <= 1))) {
    throw new Error(`Invalid archived curve: ${item.name}`);
  }
  return {...item, values};
});
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const metadata = {source_sha256: hash(sourcePath), reference_sha256: hash(referencePath),
  baseline, reference_dash_pattern: dash, legend_matches_reference: true,
  series: records.map(s => ({name: s.name, values: s.values}))};
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<desc>${escape(JSON.stringify(metadata))}</desc>
<g font-family="Times New Roman,serif" font-size="36" fill="#222222">
<text x="1188" y="78" text-anchor="middle" font-size="43">Worst source mixed Macro-F1</text>`;
for (let tick = 0; tick <= 5; tick++) {
  const value = tick / 5;
  svg += `<line x1="${left}" y1="${y(value)}" x2="${right}" y2="${y(value)}" stroke="#dddddd" stroke-width="2"/>
<text x="${left - 20}" y="${y(value) + 12}" text-anchor="end">${value.toFixed(1).replace('.0', '')}</text>`;
}
svg += `<path d="M ${left} ${top} V ${bottom} H ${right}" fill="none" stroke="#222222" stroke-width="3"/>`;
for (const epoch of [0, 10, 20, 30, 40]) {
  svg += `<line x1="${x(epoch)}" y1="${bottom}" x2="${x(epoch)}" y2="${bottom + 13}" stroke="#222222" stroke-width="3"/>
<text x="${x(epoch)}" y="${bottom + 52}" text-anchor="middle">${epoch}</text>`;
}
svg += `<text x="1188" y="993" text-anchor="middle" font-size="39">Epoch</text>`;
svg += `<line id="reference-line" x1="${x(1)}" y1="${y(baseline)}" x2="${x(40)}" y2="${y(baseline)}" ${referenceStyle}/>`;
for (const item of records) {
  svg += `<polyline points="${item.values.map(([epoch, value]) => `${x(epoch).toFixed(3)},${y(value).toFixed(3)}`).join(' ')}" fill="none" stroke="${item.color}" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>
<line x1="${item.legendX}" y1="158" x2="${item.legendX + 64}" y2="158" stroke="${item.color}" stroke-width="6.5"/>
<text x="${item.legendX + 82}" y="169" font-size="32">${item.label}</text>`;
}
svg += `<line id="reference-legend" x1="1683" y1="158" x2="1747" y2="158" ${referenceStyle}/>
<text x="1765" y="169" font-size="32">Old PGD fine40</text></g></svg>`;
const svgPath = outputPath.replace(/\.png$/i, '.svg');
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(svgPath, svg, 'utf8');
sharp(Buffer.from(svg)).png().withMetadata({density: 300}).toFile(outputPath)
  .then(info => console.log(JSON.stringify({output: outputPath, svg: svgPath, width: info.width,
    height: info.height, baseline, curve_points: records.map(s => s.values.length),
    shared_reference_dash_pattern: dash})))
  .catch(error => { console.error(error.message); process.exitCode = 1; });
