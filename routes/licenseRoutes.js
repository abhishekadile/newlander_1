const express = require('express');
const router  = express.Router();
const { seed, register, check, list, remove } = require('../controllers/licenseController');

router.post('/seed',     seed);
router.post('/register', register);
router.post('/check',    check);
router.get('/list',      list);
router.delete('/:licenseKey', remove);

module.exports = router;
