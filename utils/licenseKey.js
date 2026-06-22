function normalizeLicenseKey(key) {
  if (key == null || typeof key !== 'string') return '';
  return key.trim().toUpperCase();
}

module.exports = { normalizeLicenseKey };
