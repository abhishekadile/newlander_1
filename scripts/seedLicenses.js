/**
 * seedLicenses.js  –  local-only license key seeder
 *
 * Usage:
 *   node seedLicenses.js
 *
 * Tweak the CONFIG block below to control how many keys to create,
 * which product type to use, etc.
 */

const mongoose = require('mongoose');
require('dotenv').config({ path: require('path').resolve(__dirname, '../.env') });

// ─── CONFIG ────────────────────────────────────────────────────────────────
const MONGODB_URI = process.env.MONGODB_URI;

// Each entry: { productType, count }
const SEED_PLAN = [
  { productType: 'colony', count: 3 },
  { productType: 'cells',  count: 3 },
];
// ───────────────────────────────────────────────────────────────────────────

// Inline schema — mirrors models/License.js so we need no imports from the app
const licenseSchema = new mongoose.Schema(
  {
    licenseKey:        { type: String, required: true, unique: true, trim: true, uppercase: true },
    productType:       { type: String, required: true, trim: true },
    deviceFingerprint: { type: String, default: null },
    deviceInfo:        { type: mongoose.Schema.Types.Mixed, default: null },
    registeredAt:      { type: Date, default: null },
    downloadRedeemedAt: { type: Date, default: null },
    downloadPendingUntil: { type: Date, default: null },
  },
  { timestamps: true }
);

const License = mongoose.models.License || mongoose.model('License', licenseSchema);

// ─── Key generator ─────────────────────────────────────────────────────────
const CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no O/0/I/1 to avoid confusion

function randomSegment(len = 4) {
  let seg = '';
  for (let i = 0; i < len; i++) {
    seg += CHARS[Math.floor(Math.random() * CHARS.length)];
  }
  return seg;
}

function generateKey() {
  return [
    randomSegment(4),
    randomSegment(4),
    randomSegment(4),
    randomSegment(4),
  ].join('-');
}

// ─── Main ───────────────────────────────────────────────────────────────────
async function main() {
  console.log('Connecting to MongoDB…');
  await mongoose.connect(MONGODB_URI);
  console.log('Connected.\n');

  const inserted = [];
  const skipped  = [];

  for (const { productType, count } of SEED_PLAN) {
    console.log(`\n[${productType}] — generating ${count} keys…`);
    let added    = 0;
    let attempts = 0;

    while (added < count) {
      attempts++;

      if (attempts > count * 50) {
        console.error(`  Too many attempts for "${productType}". Aborting this batch.`);
        break;
      }

      const key    = generateKey();
      const exists = await License.exists({ licenseKey: key });

      if (exists) {
        skipped.push({ key, productType, reason: 'collision' });
        continue;
      }

      await License.create({
        licenseKey: key,
        productType,
        downloadRedeemedAt: null,
        downloadPendingUntil: null,
      });
      inserted.push({ key, productType });
      console.log(`  ✓  ${key}  (${productType})`);
      added++;
    }
  }

  console.log(`\nDone.`);
  console.log(`  Inserted : ${inserted.length}`);
  if (skipped.length) {
    console.log(`  Skipped  : ${skipped.length} (collision)`);
  }

  await mongoose.disconnect();
}

main().catch((err) => {
  console.error('Seeder error:', err.message);
  process.exit(1);
});
