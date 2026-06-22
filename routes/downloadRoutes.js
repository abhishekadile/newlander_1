const express = require('express');
const router = express.Router();
const { validateKey, streamDownload } = require('../controllers/downloadController');

router.post('/validate-key', validateKey);
router.get('/stream/:token', streamDownload);

module.exports = router;
