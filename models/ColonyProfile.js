const mongoose = require('mongoose');

const colonyProfileSchema = new mongoose.Schema(
  {
    name:        { type: String, required: true, trim: true },
    description: { type: String, default: '' },
    image_path:  { type: String, default: null },
    icon:        { type: String, default: null },
    user:        { type: mongoose.Schema.Types.ObjectId, ref: 'User', default: null },
    parameters:  [{ label: { type: String }, value: { type: String } }],
    params:      { type: mongoose.Schema.Types.Mixed, default: () => ({}) },
    source:      { type: String, enum: ['lab', 'user'], default: 'user' },
    validated:   { type: Boolean, default: false },
  },
  { timestamps: true }
);

module.exports = mongoose.model('ColonyProfile', colonyProfileSchema);
