/**
 * Environment Configuration Manager
 * =================================
 * Validates and exposes environment configuration using Zod.
 * Fails fast on startup if configuration is invalid.
 */

const { z } = require('zod');
const path = require('path');
const dotenv = require('dotenv');

// Load .env if present
dotenv.config();

const envSchema = z.object({
  PORT: z.coerce.number().int().min(1024).max(65535).default(3000),
  MONGODB_URI: z.string().default('mongodb://127.0.0.1:27017/maji_alert'),
  PYTHON_EXECUTABLE: z.string().default(
    process.env.PYTHON_EXECUTABLE || path.join(__dirname, '../../venv/bin/python')
  ),
  ML_SERVICE_PATH: z.string().default(
    process.env.ML_SERVICE_PATH || path.join(__dirname, '../../predict_service.py')
  ),
  ML_TIMEOUT_MS: z.coerce.number().int().min(5000).max(120000).default(45000),
  MAX_CONCURRENT_ML_JOBS: z.coerce.number().int().min(1).max(20).default(3),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  CORS_ORIGIN: z.string().default('*'),
  DEFAULT_YEAR: z.coerce.number().int().min(1700).max(2100).default(2026),
  RATE_LIMIT_WINDOW_MS: z.coerce.number().int().min(1000).default(60000),
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().int().min(1).default(30),
});

let parsedConfig;

try {
  parsedConfig = envSchema.parse({
    PORT: process.env.PORT,
    MONGODB_URI: process.env.MONGODB_URI,
    PYTHON_EXECUTABLE: process.env.PYTHON_EXECUTABLE,
    ML_SERVICE_PATH: process.env.ML_SERVICE_PATH,
    ML_TIMEOUT_MS: process.env.ML_TIMEOUT_MS,
    MAX_CONCURRENT_ML_JOBS: process.env.MAX_CONCURRENT_ML_JOBS,
    NODE_ENV: process.env.NODE_ENV,
    CORS_ORIGIN: process.env.CORS_ORIGIN,
    DEFAULT_YEAR: process.env.DEFAULT_YEAR,
    RATE_LIMIT_WINDOW_MS: process.env.RATE_LIMIT_WINDOW_MS,
    RATE_LIMIT_MAX_REQUESTS: process.env.RATE_LIMIT_MAX_REQUESTS,
  });
} catch (error) {
  console.error('CRITICAL: Environment variable validation failed:');
  if (error instanceof z.ZodError) {
    error.errors.forEach((err) => {
      console.error(`  - ${err.path.join('.')}: ${err.message}`);
    });
  } else {
    console.error(error);
  }
  if (process.env.NODE_ENV !== 'test') {
    process.exit(1);
  }
  // For test fallback
  parsedConfig = envSchema.parse({});
}

module.exports = parsedConfig;
