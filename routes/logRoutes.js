const express = require('express');
const router = express.Router();
const { requireAuth } = require('../middleware/authMiddleware');
const { logIngestRateLimit } = require('../middleware/logIngestRateLimit');
const { ingest } = require('../controllers/logController');

router.post('/ingest', requireAuth, logIngestRateLimit, ingest);

module.exports = router;
