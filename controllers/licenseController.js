const License = require('../models/License');
const { normalizeLicenseKey } = require('../utils/licenseKey');

/**
 * POST /api/license/seed  (admin-only)
 * Body: { adminSecret, keys: [{ licenseKey, productType }] }
 *
 * Inserts new (unregistered) license keys into the DB.
 * Skips duplicates silently.
 */
exports.seed = async (req, res) => {
  try {
    const { adminSecret, keys } = req.body;

    if (!adminSecret || adminSecret !== process.env.ADMIN_SECRET) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    if (!Array.isArray(keys) || keys.length === 0) {
      return res.status(400).json({ success: false, message: 'keys array is required.' });
    }

    const results = { inserted: [], skipped: [] };

    for (const entry of keys) {
      if (!entry.licenseKey || !entry.productType) {
        results.skipped.push({ ...entry, reason: 'missing licenseKey or productType' });
        continue;
      }

      const key = normalizeLicenseKey(entry.licenseKey);
      if (!key) {
        results.skipped.push({ ...entry, reason: 'missing licenseKey or productType' });
        continue;
      }
      const exists = await License.exists({ licenseKey: key });

      if (exists) {
        results.skipped.push({ licenseKey: key, reason: 'already exists' });
        continue;
      }

      await License.create({
        licenseKey: key,
        productType: entry.productType,
        downloadRedeemedAt: null,
        downloadPendingUntil: null,
      });
      results.inserted.push(key);
    }

    return res.status(200).json({ success: true, ...results });
  } catch (err) {
    console.error('licenseController.seed error:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

/**
 * POST /api/license/register
 * Body: { licenseKey, deviceFingerprint, deviceInfo }
 *
 * Rules:
 *  1. Key not found in DB                                      → 404 invalid key
 *  2. Key already bound to a DIFFERENT device                  → 403 key taken
 *  3. Key already bound to THIS device                         → 200 idempotent
 *  4. Key unbound BUT this device already has a key for the
 *     same productType                                         → 409 already licensed for this product
 *  5. Key unbound, device has no key for this productType      → 200 register
 */
exports.register = async (req, res) => {
  try {
    const { licenseKey, deviceFingerprint, deviceInfo } = req.body;

    if (!licenseKey || !deviceFingerprint) {
      return res.status(400).json({ success: false, message: 'licenseKey and deviceFingerprint are required.' });
    }

    const license = await License.findOne({ licenseKey: normalizeLicenseKey(licenseKey) });

    if (!license) {
      return res.status(404).json({ success: false, message: 'Invalid license key.' });
    }

    // Key already bound to a different device
    if (license.deviceFingerprint && license.deviceFingerprint !== deviceFingerprint) {
      return res.status(403).json({
        success: false,
        message: 'This license key is already registered to a different device.',
      });
    }

    // Idempotent: same device re-registering the same key
    if (license.deviceFingerprint === deviceFingerprint) {
      return res.status(200).json({
        success: true,
        message: 'Device registered successfully',
        productType: license.productType,
      });
    }

    // Device already has a different key for this product type
    const conflict = await License.findOne({
      deviceFingerprint,
      productType: license.productType,
    });

    if (conflict) {
      return res.status(409).json({
        success: false,
        message: `This device is already licensed for "${license.productType}" with a different key.`,
      });
    }

    // First-time registration for this product type on this device
    license.deviceFingerprint = deviceFingerprint;
    license.deviceInfo        = deviceInfo || null;
    license.registeredAt      = new Date();
    await license.save();

    return res.status(200).json({
      success: true,
      message: 'Device registered successfully',
      productType: license.productType,
    });
  } catch (err) {
    console.error('licenseController.register error:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

/**
 * POST /api/license/check
 * Body: { deviceFingerprint, productType? }
 *
 * With productType:    returns { isLicensed: true/false, productType }
 * Without productType: returns { isLicensed: true/false, productTypes: [...] }
 *                      isLicensed is true if the device has at least one active license.
 */
exports.check = async (req, res) => {
  try {
    const { deviceFingerprint, productType } = req.body;

    if (!deviceFingerprint) {
      return res.status(400).json({ success: false, message: 'deviceFingerprint is required.' });
    }

    // Check for a specific product type
    if (productType) {
      const license = await License.findOne({ deviceFingerprint, productType });
      return res.status(200).json({
        success: true,
        isLicensed: !!license,
        productType: license ? license.productType : null,
      });
    }

    // No productType — return all licensed types for this device
    const licenses = await License.find({ deviceFingerprint }).select('productType').lean();
    const productTypes = licenses.map(l => l.productType);

    return res.status(200).json({
      success: true,
      isLicensed: productTypes.length > 0,
      productTypes,
    });
  } catch (err) {
    console.error('licenseController.check error:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

/**
 * GET /api/license/list (admin-only)
 * Query: ?adminSecret=...
 *
 * Returns all licenses in the database.
 */
exports.list = async (req, res) => {
  try {
    const { adminSecret } = req.query;

    if (!adminSecret || adminSecret !== process.env.ADMIN_SECRET) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const licenses = await License.find().sort({ createdAt: -1 }).lean();
    return res.status(200).json({ success: true, licenses });
  } catch (err) {
    console.error('licenseController.list error:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

/**
 * DELETE /api/license/:licenseKey?adminSecret=...
 * Permanently removes a license key from the database.
 */
exports.remove = async (req, res) => {
  try {
    const { adminSecret } = req.query;
    if (!adminSecret || adminSecret !== process.env.ADMIN_SECRET) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const licenseKey = normalizeLicenseKey(req.params.licenseKey);
    if (!licenseKey) {
      return res.status(400).json({ success: false, message: 'licenseKey is required.' });
    }

    const deleted = await License.findOneAndDelete({ licenseKey });
    if (!deleted) {
      return res.status(404).json({ success: false, message: 'License key not found.' });
    }

    return res.status(200).json({ success: true, message: 'License deleted.', licenseKey });
  } catch (err) {
    console.error('licenseController.remove error:', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};
