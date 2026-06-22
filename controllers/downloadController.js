const axios = require('axios');
const License = require('../models/License');
const { normalizeLicenseKey } = require('../utils/licenseKey');

const TOKEN_TTL_MS = 60 * 1000;
const DEFAULT_GDRIVE_FILE_ID = '1v5s4-bcxOjD7n0hM_aanpgoeQLAoLNhy';

/** @type {Map<string, { fileId: string, licenseKey: string, expiresAt: number }>} */
const activeDownloads = new Map();

function generateToken(length = 12) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let token = '';
  for (let i = 0; i < length; i++) {
    token += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return token;
}

function pruneExpiredTokens() {
  const now = Date.now();
  for (const [token, entry] of activeDownloads) {
    if (now > entry.expiresAt) activeDownloads.delete(token);
  }
}

async function clearDownloadPending(licenseKey) {
  await License.updateOne(
    { licenseKey, downloadRedeemedAt: null },
    { $set: { downloadPendingUntil: null } }
  );
}

async function reserveKeyForDownload(licenseKey) {
  const pendingUntil = new Date(Date.now() + TOKEN_TTL_MS);
  const now = new Date();

  const license = await License.findOneAndUpdate(
    {
      licenseKey,
      downloadRedeemedAt: null,
      $or: [
        { downloadPendingUntil: null },
        { downloadPendingUntil: { $lte: now } },
      ],
    },
    { $set: { downloadPendingUntil: pendingUntil } },
    { new: true }
  );

  if (license) return { ok: true, license };

  const existing = await License.findOne({ licenseKey });
  if (!existing) return { ok: false, status: 404, message: 'Key not found' };
  if (existing.downloadRedeemedAt) {
    return { ok: false, status: 403, message: 'Key already redeemed' };
  }
  return { ok: false, status: 429, message: 'Download already in progress for this key' };
}

// POST /api/validate-key
async function validateKey(req, res) {
  try {
    const normalizedKey = normalizeLicenseKey(req.body?.key);
    if (!normalizedKey) {
      return res.status(400).json({ success: false, message: 'Key required' });
    }

    const reserved = await reserveKeyForDownload(normalizedKey);
    if (!reserved.ok) {
      return res.status(reserved.status).json({ success: false, message: reserved.message });
    }

    pruneExpiredTokens();

    const downloadToken = generateToken(12);
    const fileId = process.env.GDRIVE_FILE_ID || DEFAULT_GDRIVE_FILE_ID;
    const expiresAt = Date.now() + TOKEN_TTL_MS;

    activeDownloads.set(downloadToken, { fileId, licenseKey: normalizedKey, expiresAt });

    console.log(`Key ${normalizedKey} validated. Download token issued (expires in 60s).`);

    return res.status(200).json({
      success: true,
      downloadToken,
      productType: reserved.license.productType,
    });
  } catch (err) {
    console.error('[download/validate-key]', err);
    return res.status(500).json({ success: false, message: 'Internal server error' });
  }
}

async function markDownloadRedeemed(licenseKey) {
  await License.updateOne(
    { licenseKey, downloadRedeemedAt: null },
    {
      $set: {
        downloadRedeemedAt: new Date(),
        downloadPendingUntil: null,
      },
    }
  );
}

// GET /api/stream/:token
async function streamDownload(req, res) {
  const { token } = req.params;
  const entry = activeDownloads.get(token);

  if (!entry) {
    return res.status(403).json({ success: false, message: 'Invalid or expired download token' });
  }

  if (Date.now() > entry.expiresAt) {
    activeDownloads.delete(token);
    await clearDownloadPending(entry.licenseKey);
    return res.status(403).json({ success: false, message: 'Download token has expired' });
  }

  activeDownloads.delete(token);

  const apiKey = process.env.GDRIVE_API_KEY;
  if (!apiKey) {
    console.error('[download/stream] GDRIVE_API_KEY is not configured');
    await clearDownloadPending(entry.licenseKey);
    return res.status(500).json({ success: false, message: 'Download storage is not configured' });
  }

  const { fileId, licenseKey } = entry;
  const driveUrl = `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media&key=${apiKey}&acknowledgeAbuse=true`;

  try {
    console.log(`Streaming file ${fileId} for key ${licenseKey}...`);

    const driveResponse = await axios.get(driveUrl, { responseType: 'stream' });

    await markDownloadRedeemed(licenseKey);

    res.setHeader('Content-Disposition', 'attachment; filename="IncuCount_Setup.exe"');
    res.setHeader('Content-Type', 'application/octet-stream');

    driveResponse.data.pipe(res);

    driveResponse.data.on('end', () => {
      console.log(`File stream complete for key ${licenseKey}`);
    });

    driveResponse.data.on('error', async (streamErr) => {
      console.error('[download/stream] stream error:', streamErr);
      if (!res.headersSent) {
        res.status(500).json({ success: false, message: 'File stream error' });
      }
    });
  } catch (err) {
    const driveStatus = err.response?.status;
    const driveDetail = err.response?.data;
    console.error(
      '[download/stream] Google Drive error:',
      err.message,
      driveStatus ? `(HTTP ${driveStatus})` : '',
      typeof driveDetail === 'object' ? JSON.stringify(driveDetail) : driveDetail || ''
    );
    await clearDownloadPending(entry.licenseKey);
    if (!res.headersSent) {
      return res.status(500).json({ success: false, message: 'Failed to fetch file from storage' });
    }
  }
}

module.exports = { validateKey, streamDownload };
