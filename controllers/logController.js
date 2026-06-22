const DiagnosticLog = require('../models/DiagnosticLog');
const VALID_LEVELS = DiagnosticLog.VALID_LEVELS;
const MAX_ENTRIES_PER_DEVICE = DiagnosticLog.MAX_ENTRIES_PER_DEVICE;
const RETENTION_MS = DiagnosticLog.RETENTION_MS;

const MAX_BATCH = 50;
const VALID_LEVEL_SET = new Set(VALID_LEVELS);

const SENSITIVE_META_KEYS = new Set([
  'authorization',
  'password',
  'passwd',
  'token',
  'accesstoken',
  'access_token',
  'refreshtoken',
  'refresh_token',
  'licensekey',
  'license_key',
  'license',
  'apikey',
  'api_key',
  'secret',
]);

function parseTs(value) {
  if (!value) return new Date();
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? new Date() : d;
}

function normalizeLevel(level) {
  const v = String(level || 'info').toLowerCase();
  return VALID_LEVEL_SET.has(v) ? v : 'info';
}

function stripSensitiveMeta(meta) {
  if (meta == null || typeof meta !== 'object' || Array.isArray(meta)) return meta;

  const out = Array.isArray(meta) ? [] : {};
  for (const [key, value] of Object.entries(meta)) {
    const lower = key.toLowerCase();
    if (SENSITIVE_META_KEYS.has(lower)) continue;
    if (value && typeof value === 'object') {
      out[key] = stripSensitiveMeta(value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function mapEntry(e) {
  return {
    ts: parseTs(e.ts),
    level: normalizeLevel(e.level),
    event: String(e.event || 'unknown').slice(0, 128),
    message: String(e.message || '').slice(0, 2000),
    scope: e.scope != null ? String(e.scope).slice(0, 64) : undefined,
    page: e.page != null ? String(e.page).slice(0, 128) : undefined,
    sessionId: e.sessionId != null ? String(e.sessionId).slice(0, 128) : undefined,
    meta: stripSensitiveMeta(e.meta),
    receivedAt: new Date(),
  };
}

function groupByInstallId(entries) {
  const groups = new Map();
  for (const raw of entries) {
    const installId = raw.installId != null ? String(raw.installId).trim().slice(0, 128) : '';
    if (!installId) continue;
    if (!groups.has(installId)) {
      groups.set(installId, { rawSamples: [], mapped: [] });
    }
    const g = groups.get(installId);
    g.rawSamples.push(raw);
    g.mapped.push(mapEntry(raw));
  }
  return groups;
}

function latestDeviceFields(rawSamples) {
  const last = rawSamples[rawSamples.length - 1];
  return {
    appVersion: last.appVersion != null ? String(last.appVersion).slice(0, 32) : undefined,
    productType: last.productType != null ? String(last.productType).slice(0, 32) : undefined,
    platform: last.platform != null ? String(last.platform).slice(0, 64) : undefined,
  };
}

async function appendToDevice(installId, userId, rawSamples, mappedEntries) {
  const cutoff = new Date(Date.now() - RETENTION_MS);
  const deviceFields = latestDeviceFields(rawSamples);
  const lastEntryAt = mappedEntries.reduce(
    (max, e) => (e.ts > max ? e.ts : max),
    mappedEntries[0]?.ts || new Date()
  );

  await DiagnosticLog.updateOne(
    { installId },
    {
      $setOnInsert: { installId },
      $set: {
        userId,
        lastEntryAt,
        ...deviceFields,
      },
      $push: {
        entries: {
          $each: mappedEntries,
          $slice: -MAX_ENTRIES_PER_DEVICE,
        },
      },
    },
    { upsert: true }
  );

  await DiagnosticLog.updateOne(
    { installId },
    { $pull: { entries: { ts: { $lt: cutoff } } } }
  );
}

// POST /api/logs/ingest
async function ingest(req, res) {
  try {
    const entries = req.body?.entries;
    if (!Array.isArray(entries) || entries.length === 0) {
      return res.status(400).json({ success: false, message: 'entries required' });
    }
    if (entries.length > MAX_BATCH) {
      return res.status(400).json({ success: false, message: 'max 50 entries per batch' });
    }

    const groups = groupByInstallId(entries);
    if (groups.size === 0) {
      return res.status(400).json({ success: false, message: 'installId required on each entry' });
    }

    const userId = req.user.sub;
    let inserted = 0;

    for (const [installId, { rawSamples, mapped }] of groups) {
      await appendToDevice(installId, userId, rawSamples, mapped);
      inserted += mapped.length;
    }

    return res.json({
      success: true,
      inserted,
      devices: groups.size,
    });
  } catch (err) {
    console.error('[logs/ingest]', err);
    return res.status(500).json({ success: false, message: 'Failed to ingest diagnostic logs' });
  }
}

module.exports = { ingest };
