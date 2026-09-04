/**
 * Structured Logger for FRADSCR
 * =================================
 * Provides uniform log output with timestamp, log-level, and metadata.
 */

const config = require('../config/env');

const LEVELS = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const currentLevel = config.NODE_ENV === 'production' ? LEVELS.info : LEVELS.debug;

function formatLog(level, message, meta = {}) {
  const timestamp = new Date().toISOString();
  // Sanitize any potential sensitive fields in meta
  const safeMeta = { ...meta };
  ['password', 'secret', 'token', 'apiKey', 'authorization'].forEach((key) => {
    if (key in safeMeta) safeMeta[key] = '[REDACTED]';
  });

  if (config.NODE_ENV === 'production') {
    return JSON.stringify({ timestamp, level, message, ...safeMeta });
  }

  const metaStr = Object.keys(safeMeta).length > 0 ? ` | ${JSON.stringify(safeMeta)}` : '';
  return `[${timestamp}] [${level.toUpperCase()}] ${message}${metaStr}`;
}

const logger = {
  debug(message, meta) {
    if (LEVELS.debug >= currentLevel) console.debug(formatLog('debug', message, meta));
  },
  info(message, meta) {
    if (LEVELS.info >= currentLevel) console.info(formatLog('info', message, meta));
  },
  warn(message, meta) {
    if (LEVELS.warn >= currentLevel) console.warn(formatLog('warn', message, meta));
  },
  error(message, meta) {
    if (LEVELS.error >= currentLevel) console.error(formatLog('error', message, meta));
  },
};

module.exports = logger;
