const WINDOW_MS = 60 * 1000;
const MAX_REQUESTS_PER_MINUTE = 60;
const MAX_ENTRIES_PER_MINUTE = 500;

/** @type {Map<string, { requests: number[], entryBatches: { ts: number, count: number }[] }>} */
const buckets = new Map();

function getBucket(userId) {
  if (!buckets.has(userId)) {
    buckets.set(userId, { requests: [], entryBatches: [] });
  }
  return buckets.get(userId);
}

function pruneTimestamps(arr) {
  const cutoff = Date.now() - WINDOW_MS;
  while (arr.length && arr[0] < cutoff) arr.shift();
}

function pruneEntryBatches(batches) {
  const cutoff = Date.now() - WINDOW_MS;
  while (batches.length && batches[0].ts < cutoff) batches.shift();
}

function entryCountInWindow(batches) {
  return batches.reduce((sum, b) => sum + b.count, 0);
}

/**
 * Per-user rate limit: 60 requests/min, 500 entries/min.
 * Expects req.user.sub (JWT) and req.body.entries (array).
 */
function logIngestRateLimit(req, res, next) {
  const userId = req.user?.sub;
  if (!userId) return next();

  const entries = req.body?.entries;
  const batchSize = Array.isArray(entries) ? entries.length : 0;
  const bucket = getBucket(String(userId));

  pruneTimestamps(bucket.requests);
  if (bucket.requests.length >= MAX_REQUESTS_PER_MINUTE) {
    return res.status(429).json({
      success: false,
      message: 'Rate limit exceeded: max 60 ingest requests per minute',
    });
  }

  pruneEntryBatches(bucket.entryBatches);
  const entriesSoFar = entryCountInWindow(bucket.entryBatches);
  if (entriesSoFar + batchSize > MAX_ENTRIES_PER_MINUTE) {
    return res.status(429).json({
      success: false,
      message: 'Rate limit exceeded: max 500 log entries per minute',
    });
  }

  bucket.requests.push(Date.now());
  if (batchSize > 0) {
    bucket.entryBatches.push({ ts: Date.now(), count: batchSize });
  }

  next();
}

module.exports = { logIngestRateLimit };
