const mongoose = require('mongoose');

const userSchema = new mongoose.Schema(
  {
    username: { type: String, required: true, unique: true, trim: true },
    email:    { type: String, required: true, unique: true, lowercase: true, trim: true },
    password: { type: String, required: true },
    role:     { type: String, enum: ['admin', 'user'], default: 'user' },
    failedLoginAttempts: { type: Number, default: 0 },
    locked:      { type: Boolean, default: false },
    lockedUntil: { type: Date, default: null },
    lastLogin:   { type: Date, default: null },
  },
  { timestamps: true }
);

userSchema.methods.isLocked = function () {
  if (!this.locked) return false;
  if (this.lockedUntil && this.lockedUntil <= new Date()) {
    this.locked = false;
    this.lockedUntil = null;
    this.failedLoginAttempts = 0;
    return false;
  }
  return true;
};

userSchema.methods.lockAccount = function () {
  this.locked = true;
  this.lockedUntil = new Date(Date.now() + 30 * 60 * 1000); // 30 min lockout
};

userSchema.methods.incrementFailedAttempts = function () {
  this.failedLoginAttempts += 1;
  if (this.failedLoginAttempts >= 5) this.lockAccount();
};

module.exports = mongoose.model('User', userSchema);
