const mongoose = require('mongoose');

const licenseSchema = new mongoose.Schema(
  {
    licenseKey:        { type: String, required: true, unique: true, trim: true, uppercase: true },
    productType:       { type: String, required: true, trim: true },
    deviceFingerprint: { type: String, default: null },
    deviceInfo:        { type: mongoose.Schema.Types.Mixed, default: null },
    registeredAt:      { type: Date, default: null },
    downloadRedeemedAt: { type: Date, default: null },
    downloadPendingUntil: { type: Date, default: null },
  },
  { timestamps: true }
);

module.exports = mongoose.model('License', licenseSchema);
