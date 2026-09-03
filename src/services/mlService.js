/**
 * Python ML Prediction Integration Service (HTTP / FastAPI)
 * ==========================================================
 * Communicates with the persistent FastAPI ML service via HTTP fetch.
 * Completely eliminates per-request child process spawning and cold starts.
 * Features:
 * - Safe URL construction via URL and URLSearchParams
 * - Explicit AbortController timeout (ML_REQUEST_TIMEOUT_MS)
 * - Controlled transient retry policy for network glitches
 * - Strict Zod schema validation of ML responses
 * - Clear distinction of 4xx, 503 (service unavailable), 504 (timeout), and 500 errors
 */

const config = require('../config/env');
const logger = require('./logger');
const { validateMlOutput } = require('../validation/mlOutput');

class MlServiceError extends Error {
  constructor(message, clientMessage = 'Prediction service temporarily unavailable.', statusCode = 500) {
    super(message);
    this.name = 'MlServiceError';
    this.clientMessage = clientMessage;
    this.statusCode = statusCode;
  }
}

// Active request tracking for concurrency observation
let activeJobs = 0;

/**
 * Executes HTTP prediction request to persistent FastAPI ML service.
 *
 * @param {Object} params
 * @param {number} params.latitude
 * @param {number} params.longitude
 * @param {number} params.year
 * @returns {Promise<{ data: Object, durationMs: number }>}
 */
async function executePrediction({ latitude, longitude, year }) {
  activeJobs += 1;
  const startTime = Date.now();

  try {
    // Safe URL construction
    const targetUrl = new URL('/predict', config.ML_SERVICE_URL);
    targetUrl.searchParams.set('latitude', String(latitude));
    targetUrl.searchParams.set('longitude', String(longitude));
    targetUrl.searchParams.set('year', String(year));

    logger.debug('Dispatching HTTP prediction request to FastAPI', {
      url: targetUrl.origin + targetUrl.pathname,
      latitude,
      longitude,
      year,
    });

    let response;
    let attempts = 0;
    const maxAttempts = 2; // 1 initial attempt + 1 transient retry

    while (attempts < maxAttempts) {
      attempts += 1;
      const controller = new AbortController();
      const timeoutTimer = setTimeout(() => controller.abort(), config.ML_REQUEST_TIMEOUT_MS);

      try {
        response = await fetch(targetUrl.toString(), {
          method: 'GET',
          headers: {
            Accept: 'application/json',
          },
          signal: controller.signal,
        });
        clearTimeout(timeoutTimer);
        break; // Request reached server successfully
      } catch (networkErr) {
        clearTimeout(timeoutTimer);

        const isTimeout = networkErr.name === 'AbortError' || networkErr.message?.includes('aborted');
        const isNetworkError =
          networkErr.code === 'ECONNREFUSED' ||
          networkErr.message?.includes('fetch failed') ||
          networkErr.cause?.code === 'ECONNREFUSED';

        if (isTimeout) {
          logger.error('FastAPI ML request timed out', {
            timeoutMs: config.ML_REQUEST_TIMEOUT_MS,
            latitude,
            longitude,
            year,
          });
          throw new MlServiceError(
            `FastAPI request timed out after ${config.ML_REQUEST_TIMEOUT_MS}ms`,
            'Prediction request timed out. Please try again.',
            504
          );
        }

        if (isNetworkError && attempts < maxAttempts) {
          logger.warn(`FastAPI connection attempt ${attempts} failed. Retrying in 250ms...`, {
            error: networkErr.message,
          });
          await new Promise((r) => setTimeout(r, 250));
          continue;
        }

        logger.error('Failed to reach FastAPI ML service', {
          error: networkErr.message,
          url: targetUrl.toString(),
        });
        throw new MlServiceError(
          `Cannot connect to ML service at ${config.ML_SERVICE_URL}: ${networkErr.message}`,
          'Prediction service currently unavailable. Please try again later.',
          503
        );
      }
    }

    const durationMs = Date.now() - startTime;

    // Handle HTTP status codes
    if (!response.ok) {
      let errorBody = {};
      try {
        errorBody = await response.json();
      } catch (_) {
        // Non-JSON response
      }

      logger.warn('FastAPI returned non-200 response', {
        status: response.status,
        body: errorBody,
        durationMs,
      });

      if (response.status === 400 || response.status === 422) {
        throw new MlServiceError(
          `FastAPI client validation rejected: ${JSON.stringify(errorBody)}`,
          'Invalid coordinate or parameter inputs.',
          400
        );
      }

      if (response.status === 503) {
        throw new MlServiceError(
          `FastAPI service reporting not ready: ${JSON.stringify(errorBody)}`,
          'Prediction service is initializing or temporarily unavailable.',
          503
        );
      }

      throw new MlServiceError(
        `FastAPI internal error HTTP ${response.status}`,
        'Prediction service encountered an internal error.',
        response.status >= 500 ? 500 : response.status
      );
    }

    // Safely parse JSON
    let rawJson;
    try {
      rawJson = await response.json();
    } catch (parseErr) {
      logger.error('Failed to parse FastAPI JSON response', {
        error: parseErr.message,
        durationMs,
      });
      throw new MlServiceError(
        `Malformed JSON received from ML service: ${parseErr.message}`,
        'Prediction output format unrecognized.',
        502
      );
    }

    // Validate response schema using Zod
    let validatedOutput;
    try {
      validatedOutput = validateMlOutput(rawJson);
    } catch (valErr) {
      logger.error('FastAPI output failed Zod schema validation', {
        error: valErr.errors || valErr.message,
        durationMs,
      });
      throw new MlServiceError(
        'ML output schema validation failure',
        'Prediction output format unrecognized.',
        502
      );
    }

    logger.info('ML prediction completed successfully via FastAPI', {
      latitude,
      longitude,
      year,
      predictedClass: validatedOutput.predicted_drought_class,
      severity: validatedOutput.severity_label,
      durationMs,
    });

    return {
      data: validatedOutput,
      durationMs,
    };
  } finally {
    activeJobs = Math.max(0, activeJobs - 1);
  }
}

/**
 * Returns current active ML jobs count.
 */
function getActiveJobsCount() {
  return activeJobs;
}

module.exports = {
  executePrediction,
  getActiveJobsCount,
  MlServiceError,
};
