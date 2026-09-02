/**
 * Prediction Input Validation Schema (Zod)
 * ========================================
 * Enforces strict boundaries on latitude, longitude, and calendar year.
 * Rejects NaN, Infinity, -Infinity, strings, and unexpected fields.
 */

const { z } = require('zod');
const config = require('../config/env');

const predictionInputSchema = z
  .object({
    latitude: z
      .number({
        required_error: 'Latitude is required',
        invalid_type_error: 'Latitude must be a valid finite number',
      })
      .finite({ message: 'Latitude must be a finite number' })
      .min(-90.0, { message: 'Latitude must be >= -90.0' })
      .max(90.0, { message: 'Latitude must be <= 90.0' }),

    longitude: z
      .number({
        required_error: 'Longitude is required',
        invalid_type_error: 'Longitude must be a valid finite number',
      })
      .finite({ message: 'Longitude must be a finite number' })
      .min(-180.0, { message: 'Longitude must be >= -180.0' })
      .max(180.0, { message: 'Longitude must be <= 180.0' }),

    year: z
      .number({
        invalid_type_error: 'Year must be an integer',
      })
      .int({ message: 'Year must be an integer' })
      .min(1700, { message: 'Year must be >= 1700' })
      .max(2100, { message: 'Year must be <= 2100' })
      .optional()
      .default(config.DEFAULT_YEAR),
  })
  .strict({ message: 'Payload contains unexpected or prohibited fields' });

/**
 * Validates prediction input and returns parsed data or throws structured error.
 */
function validatePredictionInput(input) {
  try {
    return predictionInputSchema.parse(input);
  } catch (err) {
    if (err && !err.errors && err.issues) {
      err.errors = err.issues;
    }
    throw err;
  }
}

/**
 * Safe validation returning { success, data, error }
 */
function safeValidatePredictionInput(input) {
  const result = predictionInputSchema.safeParse(input);
  if (!result.success && result.error && !result.error.errors && result.error.issues) {
    result.error.errors = result.error.issues;
  }
  return result;
}

module.exports = {
  predictionInputSchema,
  validatePredictionInput,
  safeValidatePredictionInput,
};
