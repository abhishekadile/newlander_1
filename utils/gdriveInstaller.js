const axios = require('axios');

const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;

/** @type {{ fileId: string, fileName: string, expiresAt: number } | null} */
let installerCache = null;

function isExeFile(name) {
  return typeof name === 'string' && name.toLowerCase().endsWith('.exe');
}

async function listInstallerCandidates(apiKey, folderId) {
  const q = [
    `'${folderId}' in parents`,
    'trashed = false',
    "mimeType != 'application/vnd.google-apps.folder'",
  ].join(' and ');

  const { data } = await axios.get('https://www.googleapis.com/drive/v3/files', {
    params: {
      q,
      orderBy: 'modifiedTime desc',
      fields: 'files(id,name,modifiedTime)',
      pageSize: 50,
      key: apiKey,
    },
  });

  return (data.files || []).filter((f) => isExeFile(f.name));
}

/**
 * Returns the newest .exe in a public Drive folder. Result is cached briefly.
 * @returns {Promise<{ fileId: string, fileName: string } | null>}
 */
async function resolveLatestInstaller(apiKey, folderId) {
  const cacheTtl =
    Number.parseInt(process.env.GDRIVE_CACHE_TTL_MS, 10) || DEFAULT_CACHE_TTL_MS;
  const now = Date.now();

  if (
    installerCache &&
    installerCache.expiresAt > now &&
    installerCache.folderId === folderId
  ) {
    return { fileId: installerCache.fileId, fileName: installerCache.fileName };
  }

  const candidates = await listInstallerCandidates(apiKey, folderId);
  if (candidates.length === 0) return null;

  const latest = candidates[0];
  installerCache = {
    folderId,
    fileId: latest.id,
    fileName: latest.name,
    expiresAt: now + cacheTtl,
  };

  return { fileId: latest.id, fileName: latest.name };
}

module.exports = { resolveLatestInstaller, isExeFile };
