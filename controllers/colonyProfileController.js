const mongoose = require('mongoose');
const ColonyProfile = require('../models/ColonyProfile');
const {
  buildProfileParams,
  presetFromDetection,
  validateProfileParams,
  normalizeProfile,
} = require('../utils/colonyProfilePreset');

// ---------------------------------------------------------------------------
// POST /api/colony-profiles/seed  (admin-only)
// Body: { adminSecret }  — inserts the 4 built-in global profiles if absent
// ---------------------------------------------------------------------------
const DEFAULT_PROFILES = [
  {
    name: 'Anarobic Count Film',
    description: 'Used for detecting and quantifying anaerobic bacteria that grow in oxygen-free environments.',
    image_path: '../../assets/public/Trays/AC.png',
    icon: 'anaerobic',
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Incubation', value: '37°C' },
      { label: 'Duration',   value: '48 hrs' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'Coliform (CC) Film',
    description: 'Designed to detect coliform bacteria, commonly used for testing water and food safety.',
    image_path: '../../assets/public/Trays/CF.png',
    icon: 'coliform',
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Incubation', value: '35°C' },
      { label: 'Duration',   value: '24 hrs' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'MacConkey Plates',
    description: 'Selective medium for isolating and differentiating Gram-negative bacteria based on lactose fermentation.',
    image_path: '../../assets/public/Trays/MacConkey.png',
    icon: 'maconkey',
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Medium',    value: 'Agar Plate' },
      { label: 'Indicator', value: 'Neutral Red' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: true,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'Nutrient Plates',
    description: 'General-purpose medium supporting the growth of a wide range of non-fastidious organisms.',
    image_path: '../../assets/public/Trays/NP.png',
    icon: 'nutrient',
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Medium',   value: 'Agar Plate' },
      { label: 'Use case', value: 'General growth' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 5,
      min_radius: 4,
      max_radius: 185,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
];

function resolveSource(source) {
  return source === 'lab' ? 'lab' : 'user';
}

exports.seed = async (req, res) => {
  try {
    const { adminSecret } = req.body;
    if (!adminSecret || adminSecret !== process.env.ADMIN_SECRET) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const results = { inserted: [], skipped: [] };

    for (const data of DEFAULT_PROFILES) {
      const exists = await ColonyProfile.exists({ name: data.name });
      if (exists) {
        results.skipped.push(data.name);
      } else {
        await ColonyProfile.create({ ...data, user: null });
        results.inserted.push(data.name);
      }
    }

    return res.json({ success: true, ...results });
  } catch (err) {
    console.error('[colony-profiles/seed]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

function checkAdminSecret(adminSecret) {
  return adminSecret && adminSecret === process.env.ADMIN_SECRET;
}

// ---------------------------------------------------------------------------
// GET /api/colony-profiles/admin/list?adminSecret=...
// Admin-only — all profiles for review on the Render portal
// ---------------------------------------------------------------------------
exports.adminList = async (req, res) => {
  try {
    if (!checkAdminSecret(req.query.adminSecret)) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const profiles = await ColonyProfile.find()
      .sort({ createdAt: -1 })
      .lean();

    return res.json({
      success: true,
      profiles: profiles.map(normalizeProfile),
    });
  } catch (err) {
    console.error('[colony-profiles/admin/list]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// PATCH /api/colony-profiles/admin/:id/validation
// Body: { adminSecret, validated: boolean }
// Approve or revoke a profile for global sync
// ---------------------------------------------------------------------------
exports.adminSetValidation = async (req, res) => {
  try {
    const { adminSecret, validated } = req.body;
    if (!checkAdminSecret(adminSecret)) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: 'Invalid profile id.' });
    }

    if (typeof validated !== 'boolean') {
      return res.status(400).json({ success: false, message: 'validated must be a boolean.' });
    }

    const profile = await ColonyProfile.findById(id);
    if (!profile) {
      return res.status(404).json({ success: false, message: 'Profile not found.' });
    }

    profile.validated = validated;
    await profile.save();

    return res.json({ success: true, profile: normalizeProfile(profile) });
  } catch (err) {
    console.error('[colony-profiles/admin/validation]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// DELETE /api/colony-profiles/admin/:id?adminSecret=...
// Admin-only — permanently remove a profile from the central catalog
// ---------------------------------------------------------------------------
exports.adminRemove = async (req, res) => {
  try {
    if (!checkAdminSecret(req.query.adminSecret)) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: 'Invalid profile id.' });
    }

    const profile = await ColonyProfile.findByIdAndDelete(id);
    if (!profile) {
      return res.status(404).json({ success: false, message: 'Profile not found.' });
    }

    return res.json({ success: true, id });
  } catch (err) {
    console.error('[colony-profiles/admin/remove]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// GET /api/colony-profiles?userId=<id>
// No auth required (called from main process / DatabaseManager)
// ---------------------------------------------------------------------------
exports.list = async (req, res) => {
  try {
    const filter = {};
    if (req.query.userId) {
      if (mongoose.Types.ObjectId.isValid(req.query.userId)) {
        const userOid = new mongoose.Types.ObjectId(req.query.userId);
        // Own profiles (any validation state) + everyone else's only when approved
        filter.$or = [
          { user: userOid },
          { validated: true },
        ];
      }
    } else if (req.query.validatedOnly === 'true') {
      filter.validated = true;
    }

    const profiles = await ColonyProfile.find(filter).lean();

    return res.json({
      success: true,
      profiles: profiles.map(normalizeProfile),
    });
  } catch (err) {
    console.error('[colony-profiles/list]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// POST /api/colony-profiles
// No auth required (called from main process / DatabaseManager)
// ---------------------------------------------------------------------------
exports.create = async (req, res) => {
  try {
    const {
      name,
      description,
      image_path,
      icon,
      parameters,
      params,
      userId,
      source,
      validated,
    } = req.body;

    if (!name) {
      return res.status(400).json({ success: false, message: 'name is required.' });
    }

    const validation = validateProfileParams(params);
    if (!validation.valid) {
      return res.status(400).json({ success: false, message: validation.message });
    }

    const profileData = {
      name,
      description: description || '',
      image_path:  image_path  || null,
      icon:        icon        || null,
      parameters:  parameters  || [],
      params:      buildProfileParams({ ...req.body, params: validation.params }),
      source:      resolveSource(source),
      validated:   validated === true,
      user: (userId && mongoose.Types.ObjectId.isValid(userId))
        ? new mongoose.Types.ObjectId(userId)
        : null,
    };

    const profile = await ColonyProfile.create(profileData);

    return res.status(201).json({ success: true, profile: normalizeProfile(profile) });
  } catch (err) {
    console.error('[colony-profiles/create]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// PUT /api/colony-profiles/:id
// Requires Bearer token (called from renderer via api.js)
// ---------------------------------------------------------------------------
exports.update = async (req, res) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: 'Invalid profile id.' });
    }

    const { userId, description, parameters, params, source, validated } = req.body;

    const profile = await ColonyProfile.findById(id);
    if (!profile) {
      return res.status(404).json({ success: false, message: 'Profile not found.' });
    }

    const isOwner  = profile.user && profile.user.toString() === (userId || req.user?.sub);
    const isAdmin  = req.user?.role === 'admin';
    const isGlobal = profile.user == null;

    if (!isOwner && !isAdmin && !isGlobal) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    if (description !== undefined) profile.description = description;
    if (parameters  !== undefined) profile.parameters  = parameters;

    if (params !== undefined) {
      const validation = validateProfileParams(params);
      if (!validation.valid) {
        return res.status(400).json({ success: false, message: validation.message });
      }
      profile.params = buildProfileParams({ ...req.body, params: validation.params });
    }

    if (source !== undefined) profile.source = resolveSource(source);
    if (validated !== undefined) profile.validated = validated === true;

    await profile.save();

    return res.json({ success: true, profile: normalizeProfile(profile) });
  } catch (err) {
    console.error('[colony-profiles/update]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};

// ---------------------------------------------------------------------------
// DELETE /api/colony-profiles/:id?userId=<id>
// Requires Bearer token (called from renderer via api.js)
// ---------------------------------------------------------------------------
exports.remove = async (req, res) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: 'Invalid profile id.' });
    }

    const userId = req.query.userId || req.body.userId;

    const profile = await ColonyProfile.findById(id);
    if (!profile) {
      return res.status(404).json({ success: false, message: 'Profile not found.' });
    }

    const isOwner = profile.user && profile.user.toString() === (userId || req.user?.sub);
    const isAdmin = req.user?.role === 'admin';

    if (!isOwner && !isAdmin) {
      return res.status(403).json({ success: false, message: 'Forbidden.' });
    }

    await profile.deleteOne();

    return res.json({ success: true });
  } catch (err) {
    console.error('[colony-profiles/remove]', err);
    return res.status(500).json({ success: false, message: 'Internal server error.' });
  }
};
