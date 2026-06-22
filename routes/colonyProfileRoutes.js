const express      = require('express');
const router       = express.Router();
const { requireAuth } = require('../middleware/authMiddleware');
const {
  seed,
  adminList,
  adminSetValidation,
  adminRemove,
  list,
  create,
  update,
  remove,
} = require('../controllers/colonyProfileController');

// Admin seed endpoint
router.post('/seed', seed);

// Admin portal (Render dashboard)
router.get('/admin/list', adminList);
router.patch('/admin/:id/validation', adminSetValidation);
router.delete('/admin/:id', adminRemove);

// No auth — called from Electron main process (DatabaseManager)
router.get('/',     list);
router.post('/',    create);

// Auth required — called from renderer via api.js
router.put('/:id',    requireAuth, update);
router.delete('/:id', requireAuth, remove);

module.exports = router;
