const DETECTION_KEYS = [
  'threshold_type',
  'threshold_value',
  'min_radius',
  'max_radius',
  'enable_color_grouping',
  'coarseness',
  'neighbours',
];

const DEFAULT_DETECTION = {
  threshold_type: 'regular',
  threshold_value: 15,
  min_radius: 3,
  max_radius: 50,
  enable_color_grouping: false,
  coarseness: 10,
  neighbours: 10,
};

const DEFAULT_CAMERA = {
  brightness: 0,
  exposure: 0,
  contrast: 0,
  device_label_hint: null,
};

const DEFAULT_LIGHTING = {
  relays: [false, false, false, false, true, true, false, false],
};

const DEFAULT_REGION = {
  counting_circle: {
    center_norm: { x: 0.5, y: 0.5 },
    radius_norm: 0.42,
  },
  zone_masks: {
    include: [],
    exclude: [],
  },
  use_on_live_guide: true,
};

const DEFAULT_WORKFLOW = {
  primary_action: 'count',
};

const DEFAULT_LOCKS = {
  camera: true,
  lighting: true,
  detection_sliders: false,
  region: false,
};

function pickDetection(source) {
  if (!source || typeof source !== 'object') return {};
  const picked = {};
  for (const key of DETECTION_KEYS) {
    if (source[key] !== undefined) picked[key] = source[key];
  }
  return picked;
}

function isFlatLegacyParams(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return false;
  if (raw.schema_version != null) return false;
  return DETECTION_KEYS.some((key) => Object.prototype.hasOwnProperty.call(raw, key));
}

function getDefaultPreset() {
  return {
    schema_version: 1,
    detection: { ...DEFAULT_DETECTION },
    camera: { ...DEFAULT_CAMERA },
    lighting: {
      relays: [...DEFAULT_LIGHTING.relays],
    },
    region: {
      counting_circle: {
        center_norm: { ...DEFAULT_REGION.counting_circle.center_norm },
        radius_norm: DEFAULT_REGION.counting_circle.radius_norm,
      },
      zone_masks: {
        include: [...DEFAULT_REGION.zone_masks.include],
        exclude: [...DEFAULT_REGION.zone_masks.exclude],
      },
      use_on_live_guide: DEFAULT_REGION.use_on_live_guide,
    },
    workflow: { ...DEFAULT_WORKFLOW },
    locks: { ...DEFAULT_LOCKS },
  };
}

function mergeRegion(defaults, incoming) {
  if (!incoming) return defaults;
  return {
    counting_circle: {
      center_norm: {
        x: incoming.counting_circle?.center_norm?.x ?? defaults.counting_circle.center_norm.x,
        y: incoming.counting_circle?.center_norm?.y ?? defaults.counting_circle.center_norm.y,
      },
      radius_norm: incoming.counting_circle?.radius_norm ?? defaults.counting_circle.radius_norm,
    },
    zone_masks: {
      include: Array.isArray(incoming.zone_masks?.include)
        ? [...incoming.zone_masks.include]
        : [...defaults.zone_masks.include],
      exclude: Array.isArray(incoming.zone_masks?.exclude)
        ? [...incoming.zone_masks.exclude]
        : [...defaults.zone_masks.exclude],
    },
    use_on_live_guide: incoming.use_on_live_guide ?? defaults.use_on_live_guide,
  };
}

function normalizeVersionedPreset(raw) {
  const defaults = getDefaultPreset();
  return {
    schema_version: raw.schema_version ?? defaults.schema_version,
    detection: { ...defaults.detection, ...(raw.detection || {}), ...pickDetection(raw) },
    camera: { ...defaults.camera, ...(raw.camera || {}) },
    lighting: {
      relays: Array.isArray(raw.lighting?.relays)
        ? [...raw.lighting.relays]
        : [...defaults.lighting.relays],
    },
    region: mergeRegion(defaults.region, raw.region),
    workflow: { ...defaults.workflow, ...(raw.workflow || {}) },
    locks: raw.locks ? { ...defaults.locks, ...raw.locks } : { ...defaults.locks },
  };
}

/** Wrap flat legacy params or fill defaults for missing versioned sections. */
function parseProfilePreset(rawParams) {
  const raw = rawParams && typeof rawParams === 'object' && !Array.isArray(rawParams)
    ? rawParams
    : {};

  if (raw.schema_version != null || raw.detection) {
    return normalizeVersionedPreset(raw);
  }

  if (isFlatLegacyParams(raw)) {
    return normalizeVersionedPreset({
      schema_version: 1,
      detection: pickDetection(raw),
    });
  }

  return getDefaultPreset();
}

/** Build stored params from flat Electron fields and/or versioned preset sections. */
function buildProfileParams(input = {}) {
  const base = parseProfilePreset(input.params);

  const flatDetection = pickDetection(input);
  if (Object.keys(flatDetection).length) {
    base.detection = { ...base.detection, ...flatDetection };
  }

  for (const section of ['camera', 'lighting', 'region', 'workflow', 'locks']) {
    const fromParams = input.params?.[section];
    const fromTop = input[section];

    if (section === 'region') {
      if (fromParams) base.region = mergeRegion(base.region, fromParams);
      if (fromTop) base.region = mergeRegion(base.region, fromTop);
    } else if (section === 'lighting') {
      if (fromParams?.relays) base.lighting = { relays: [...fromParams.relays] };
      if (fromTop?.relays) base.lighting = { relays: [...fromTop.relays] };
    } else {
      if (fromParams) base[section] = { ...base[section], ...fromParams };
      if (fromTop) base[section] = { ...base[section], ...fromTop };
    }
  }

  if (input.params?.schema_version != null) {
    base.schema_version = input.params.schema_version;
  }

  return base;
}

function getDetectionParams(preset) {
  return parseProfilePreset(preset).detection;
}

function presetFromDetection(detectionOverrides = {}) {
  const preset = getDefaultPreset();
  preset.detection = { ...preset.detection, ...detectionOverrides };
  return preset;
}

function validateProfileParams(rawParams) {
  if (rawParams == null) {
    return { valid: true, params: getDefaultPreset() };
  }

  if (typeof rawParams !== 'object' || Array.isArray(rawParams)) {
    return { valid: false, message: 'params must be an object.' };
  }

  if (rawParams.schema_version != null && rawParams.schema_version !== 1) {
    return { valid: false, message: 'Unsupported schema_version.' };
  }

  const hasDetectionBlock = rawParams.detection && typeof rawParams.detection === 'object';
  if (!hasDetectionBlock && !isFlatLegacyParams(rawParams)) {
    return { valid: false, message: 'params must include a detection block or legacy detection fields.' };
  }

  const preset = parseProfilePreset(rawParams);
  const d = preset.detection;
  const errors = [];

  if (typeof d.threshold_type !== 'string' || !d.threshold_type.trim()) {
    errors.push('detection.threshold_type is required.');
  }
  if (typeof d.threshold_value !== 'number' || Number.isNaN(d.threshold_value)) {
    errors.push('detection.threshold_value must be a number.');
  }
  if (typeof d.min_radius !== 'number' || Number.isNaN(d.min_radius)) {
    errors.push('detection.min_radius must be a number.');
  }
  if (typeof d.max_radius !== 'number' || Number.isNaN(d.max_radius)) {
    errors.push('detection.max_radius must be a number.');
  }
  if (d.min_radius > d.max_radius) {
    errors.push('detection.min_radius must be less than or equal to max_radius.');
  }
  if (typeof d.enable_color_grouping !== 'boolean') {
    errors.push('detection.enable_color_grouping must be a boolean.');
  }
  if (typeof d.coarseness !== 'number' || Number.isNaN(d.coarseness)) {
    errors.push('detection.coarseness must be a number.');
  }
  if (typeof d.neighbours !== 'number' || Number.isNaN(d.neighbours)) {
    errors.push('detection.neighbours must be a number.');
  }

  if (errors.length) {
    return { valid: false, message: errors.join(' ') };
  }

  return { valid: true, params: preset };
}

/** API response shape: full versioned params plus flat legacy detection fields. */
function normalizeProfile(profile) {
  const plain = profile?.toObject ? profile.toObject() : { ...profile };
  const preset = parseProfilePreset(plain.params);
  const d = preset.detection;

  return {
    ...plain,
    params: preset,
    threshold_type: d.threshold_type,
    threshold_value: d.threshold_value,
    min_radius: d.min_radius,
    max_radius: d.max_radius,
    enable_color_grouping: d.enable_color_grouping,
    coarseness: d.coarseness,
    neighbours: d.neighbours,
  };
}

module.exports = {
  DETECTION_KEYS,
  DEFAULT_DETECTION,
  getDefaultPreset,
  parseProfilePreset,
  buildProfileParams,
  getDetectionParams,
  presetFromDetection,
  validateProfileParams,
  normalizeProfile,
};
