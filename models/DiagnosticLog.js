const mongoose = require('mongoose');

const VALID_LEVELS = ['debug', 'info', 'warn', 'error'];

const logEntrySchema = new mongoose.Schema(
  {
    ts: { type: Date, required: true },
    level: { type: String, required: true },
    event: { type: String, required: true },
    message: { type: String, default: '' },
    scope: String,
    page: String,
    sessionId: String,
    meta: { type: mongoose.Schema.Types.Mixed },
    receivedAt: { type: Date, default: Date.now },
  },
  { _id: true }
);

const deviceDiagnosticLogSchema = new mongoose.Schema(
  {
    installId: { type: String, required: true, unique: true, index: true },
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', index: true },
    appVersion: String,
    productType: String,
    platform: String,
    lastEntryAt: { type: Date, index: true },
    entries: { type: [logEntrySchema], default: [] },
  },
  { collection: 'diagnostic_logs', timestamps: true }
);

deviceDiagnosticLogSchema.index({ userId: 1, lastEntryAt: -1 });
deviceDiagnosticLogSchema.index({ 'entries.event': 1, lastEntryAt: -1 });

module.exports = mongoose.model('DiagnosticLog', deviceDiagnosticLogSchema);
module.exports.VALID_LEVELS = VALID_LEVELS;
module.exports.MAX_ENTRIES_PER_DEVICE = 10000;
module.exports.RETENTION_MS = 90 * 24 * 3600 * 1000;
