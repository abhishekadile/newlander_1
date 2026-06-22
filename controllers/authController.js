const bcrypt = require('bcrypt');
const jwt    = require('jsonwebtoken');
const User   = require('../models/User');

const JWT_SECRET     = process.env.JWT_SECRET     || 'change-me-in-production';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

function makeToken(user) {
  return jwt.sign({ sub: user._id, role: user.role }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

function safeUser(user) {
  return { id: user._id, username: user.username, email: user.email, role: user.role };
}

// POST /auth/register  { username, email, password }
async function register(req, res) {
  try {
    const { username, email, password } = req.body;
    if (!username || !email || !password)
      return res.status(400).json({ success: false, message: 'username, email and password are required' });

    const exists = await User.findOne({ $or: [{ username }, { email }] });
    if (exists)
      return res.status(409).json({ success: false, message: 'Username or email already taken' });

    const hash = await bcrypt.hash(password, 12);
    const user = await User.create({ username, email, password: hash });
    const token = makeToken(user);

    return res.status(201).json({ success: true, userData: safeUser(user), token });
  } catch (err) {
    console.error('[auth/register]', err);
    return res.status(500).json({ success: false, message: 'Server error during registration' });
  }
}

// POST /auth/login  { usernameOrEmail, password }
async function login(req, res) {
  try {
    const { usernameOrEmail, password } = req.body;
    if (!usernameOrEmail || !password)
      return res.status(400).json({ success: false, message: 'Credentials are required' });

    const user = await User.findOne({
      $or: [{ username: usernameOrEmail }, { email: usernameOrEmail }],
    });

    if (!user)
      return res.status(401).json({ success: false, message: 'Invalid credentials' });

    if (user.isLocked()) {
      const minutesLeft = Math.ceil((user.lockedUntil - Date.now()) / 60000);
      await user.save();
      return res.status(423).json({
        success: false,
        message: `Account locked. Try again in ${minutesLeft} minute(s)`,
      });
    }

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) {
      user.incrementFailedAttempts();
      await user.save();
      const remaining = 5 - user.failedLoginAttempts;
      const msg = remaining > 0
        ? `Invalid credentials. ${remaining} attempt(s) remaining`
        : 'Account locked due to too many failed attempts';
      return res.status(401).json({ success: false, message: msg });
    }

    // Successful login — reset counters
    user.failedLoginAttempts = 0;
    user.locked      = false;
    user.lockedUntil = null;
    user.lastLogin   = new Date();
    await user.save();

    const token = makeToken(user);
    return res.json({ success: true, userData: safeUser(user), token });
  } catch (err) {
    console.error('[auth/login]', err);
    return res.status(500).json({ success: false, message: 'Server error during login' });
  }
}

// POST /auth/logout  { userId }
async function logout(req, res) {
  // JWT is stateless — client drops the token; nothing to invalidate server-side
  return res.json({ success: true, message: 'Logged out' });
}

// GET /auth/me   (requires Authorization: Bearer <token> header)
async function getMe(req, res) {
  try {
    const token = (req.headers.authorization || '').replace('Bearer ', '');
    if (!token)
      return res.status(401).json({ success: false, message: 'No token provided' });

    let payload;
    try {
      payload = jwt.verify(token, JWT_SECRET);
    } catch {
      return res.status(401).json({ success: false, message: 'Invalid or expired token' });
    }

    const user = await User.findById(payload.sub).select('-password');
    if (!user)
      return res.status(404).json({ success: false, message: 'User not found' });

    return res.json({ success: true, userData: safeUser(user) });
  } catch (err) {
    console.error('[auth/me]', err);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
}

module.exports = { register, login, logout, getMe };
