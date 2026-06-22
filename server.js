require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs-extra');
const cors = require('cors');
const mongoose = require('mongoose');
const ColonyDetector = require('./colonyDetector');

// Connect to MongoDB
mongoose
  .connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/incucount')
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Parse JSON bodies
app.use(express.json({ limit: '1mb' }));

// Auth routes
const authRoutes = require('./routes/authRoutes');
app.use('/api/auth', authRoutes);

// License routes
const licenseRoutes = require('./routes/licenseRoutes');
app.use('/api/license', licenseRoutes);

// Colony profile routes
const colonyProfileRoutes = require('./routes/colonyProfileRoutes');
app.use('/api/colony-profiles', colonyProfileRoutes);

// Diagnostic log ingest (desktop app support/debug)
const logRoutes = require('./routes/logRoutes');
app.use('/api/logs', logRoutes);

// Website installer download (license key → one-time stream token)
const downloadRoutes = require('./routes/downloadRoutes');
app.use('/api', downloadRoutes);

// Health check
app.get('/health', (req, res) => res.json({ success: true, status: 'ok' }));

// Log all requests
app.use((req, res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
});

// Ensure uploads directory exists
const UPLOADS_DIR = path.join(__dirname, 'uploads');
fs.ensureDirSync(UPLOADS_DIR);

// Known sample images (local-only) referenced by image_id
const IMAGES_DIR = path.join(__dirname, 'images');
const IMAGE_ID_TO_FILENAME = {
    // IDs used by Shopify section
    '82.bmp': '82.bmp',
    '85.bmp': '85.bmp',
    'standard1': 'standard 1.jpg',
    'complex1': 'complex 1.jpg',
    'WIN_20250905_11_49_20_Pro': 'WIN_20250905_11_49_20_Pro.jpg',
    'WIN_20250905_11_48_18_Pro': 'WIN_20250905_11_48_18_Pro.jpg',
    'WIN_20250905_11_42_42_Pro': 'WIN_20250905_11_42_42_Pro.jpg',

    // Support a filename mismatch (backend has 11_44_26, Shopify URL says 11_44_26)
    'WIN_20250905_11_44_26_Pro': 'WIN_20250905_11_44_26_Pro.jpg',
    'WIN_20250905_11_44_26_Pro': 'WIN_20250905_11_44_26_Pro.jpg'
};

function resolveKnownImagePath(imageId) {
    const filename = IMAGE_ID_TO_FILENAME[String(imageId || '')] || null;
    if (!filename) return null;

    const base = path.resolve(IMAGES_DIR) + path.sep;
    const resolved = path.resolve(IMAGES_DIR, filename);

    // Prevent path traversal even if mapping is changed in the future
    if (!resolved.startsWith(base)) return null;
    return resolved;
}

function parseBool(v) {
    if (v === true) return true;
    if (v === false) return false;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') return v.toLowerCase() === 'true' || v === '1';
    return false;
}

function parseNum(v, fallback) {
    const n = typeof v === 'number' ? v : (v === '' || v == null ? NaN : Number(v));
    return Number.isFinite(n) ? n : fallback;
}

// Configure storage
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, UPLOADS_DIR)
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9)
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname))
    }
});

const upload = multer({ storage: storage });

const detector = new ColonyDetector();

// Routes
// Routes
// Static files are served from 'public' directory

app.get('/images/catalog', (req, res) => {
    res.json({
        success: true,
        images: Object.keys(IMAGE_ID_TO_FILENAME).map((id) => ({
            id,
            filename: IMAGE_ID_TO_FILENAME[id]
        }))
    });
});

app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        // Backwards compatible behavior:
        // - If a file is uploaded (multipart), use it
        // - Else expect JSON body with image_id (local file)
        let imagePath = null;

        if (req.file && req.file.path) {
            imagePath = req.file.path;
        } else {
            const imageId = req.body && req.body.image_id;
            console.log('imageId', imageId);
            if (!imageId) {
                return res.status(400).json({ success: false, error: 'No image uploaded and no image_id provided.' });
            }

            imagePath = resolveKnownImagePath(imageId);
            console.log('imagePath', imagePath);
            if (!imagePath) {
                return res.status(400).json({ success: false, error: `Unknown image_id: ${imageId}` });
            }

            const exists = await fs.pathExists(imagePath);
            if (!exists) {
                return res.status(400).json({ success: false, error: `Image not found on server for image_id: ${imageId}` });
            }
        }
        
        // Parse detection params from body
        const params = {
            threshold_type: req.body.threshold_type || 'regular',
            threshold_value: parseNum(req.body.threshold_value, 15),
            min_radius: parseNum(req.body.min_radius, 3),
            max_radius: parseNum(req.body.max_radius, 50),
            enable_color_grouping: parseBool(req.body.enable_color_grouping),
            coarseness: parseNum(req.body.coarseness, 10.0),
            neighbours: parseNum(req.body.neighbours, 10)
        };
        console.log('params', params);

        const result = await detector.detectColonies(imagePath, params);

        if (result.success) {
            res.json(result);
        } else {
            res.status(500).json(result);
        }

        // Cleanup uploaded file immediately if not needed for debugging?
        // For now, let's keep it or maybe clean it up. 
        // Best practice to clean up:
        // await fs.remove(imagePath).catch(console.error); 

    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

app.use('/uploads', express.static(UPLOADS_DIR));
app.use(express.static('public'));

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
