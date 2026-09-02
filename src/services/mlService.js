/**
 * Python ML Prediction Integration Service
 * ========================================
 * Safely executes `predict_service.py` via `child_process.spawn` (non-shell).
 * Provides concurrency gating, execution timeouts, orphan process mitigation,
 * runtime output validation with Zod, and safe error masking.
 */

const childProcess = require('child_process');
const path = require('path');
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

// Global active job counter for concurrency limiting
let activeJobs = 0;

/**
 * Executes Python drought prediction CLI safely using an argument array.
 *
 * @param {Object} params
 * @param {number} params.latitude
 * @param {number} params.longitude
 * @param {number} params.year
 * @returns {Promise<Object>} Validated prediction result and execution metadata
 */
async function executePrediction({ latitude, longitude, year }) {
  // 1. Concurrency control
  if (activeJobs >= config.MAX_CONCURRENT_ML_JOBS) {
    logger.warn('ML execution concurrency capacity reached', {
      activeJobs,
      limit: config.MAX_CONCURRENT_ML_JOBS,
    });
    throw new MlServiceError(
      `Concurrency capacity exceeded (${activeJobs}/${config.MAX_CONCURRENT_ML_JOBS})`,
      'Prediction service is at peak capacity. Please retry in a few seconds.',
      503
    );
  }

  activeJobs += 1;
  const startTime = Date.now();

  const pythonBin = config.PYTHON_EXECUTABLE;
  const scriptPath = config.ML_SERVICE_PATH;

  // Strict argument array (prevents any command injection)
  const args = [
    scriptPath,
    '--lat',
    String(latitude),
    '--lon',
    String(longitude),
    '--year',
    String(year),
  ];

  logger.debug('Spawning Python ML subprocess', {
    executable: pythonBin,
    script: scriptPath,
    latitude,
    longitude,
    year,
  });

  return new Promise((resolve, reject) => {
    let stdoutBuffer = '';
    let stderrBuffer = '';
    let isSettled = false;
    let timeoutTimer = null;

    const child = childProcess.spawn(pythonBin, args, {
      shell: false, // MANDATORY: Disallow shell interpolation
      windowsHide: true,
      cwd: path.dirname(scriptPath),
    });

    const cleanup = () => {
      if (timeoutTimer) {
        clearTimeout(timeoutTimer);
        timeoutTimer = null;
      }
      activeJobs = Math.max(0, activeJobs - 1);
    };

    // 2. Timeout safeguard
    timeoutTimer = setTimeout(() => {
      if (isSettled) return;
      isSettled = true;
      logger.error('ML subprocess timed out', {
        timeoutMs: config.ML_TIMEOUT_MS,
        latitude,
        longitude,
        year,
      });

      // Terminate subprocess tree
      child.kill('SIGTERM');
      setTimeout(() => {
        if (!child.killed) child.kill('SIGKILL');
      }, 2000);

      cleanup();
      reject(
        new MlServiceError(
          `Subprocess timed out after ${config.ML_TIMEOUT_MS}ms`,
          'Prediction request timed out. Please try again later.',
          504
        )
      );
    }, config.ML_TIMEOUT_MS);

    // 3. Stream handlers
    child.stdout.on('data', (chunk) => {
      stdoutBuffer += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString();
    });

    child.on('error', (err) => {
      if (isSettled) return;
      isSettled = true;
      cleanup();
      logger.error('Failed to spawn Python process', { error: err.message });
      reject(
        new MlServiceError(
          `Process spawn failed: ${err.message}`,
          'Prediction engine could not be started.'
        )
      );
    });

    child.on('close', (code, signal) => {
      if (isSettled) return;
      isSettled = true;
      cleanup();

      const durationMs = Date.now() - startTime;

      if (code !== 0) {
        logger.error('Python ML subprocess returned non-zero exit code', {
          exitCode: code,
          signal,
          stderr: stderrBuffer.slice(0, 1000),
          durationMs,
        });
        return reject(
          new MlServiceError(
            `Subprocess exited with code ${code}`,
            'Prediction service temporarily unavailable.'
          )
        );
      }

      const trimmedStdout = stdoutBuffer.trim();
      if (!trimmedStdout) {
        logger.error('Python ML subprocess produced empty stdout', { durationMs });
        return reject(
          new MlServiceError('Empty output received from ML service', 'No prediction data received.')
        );
      }

      // 4. Safe JSON parsing
      let parsedJson;
      try {
        parsedJson = JSON.parse(trimmedStdout);
      } catch (parseErr) {
        logger.error('Failed to parse ML JSON stdout', {
          error: parseErr.message,
          rawOutput: trimmedStdout.slice(0, 500),
          durationMs,
        });
        return reject(
          new MlServiceError(
            `Malformed JSON output: ${parseErr.message}`,
            'Invalid response from prediction engine.'
          )
        );
      }

      // 5. Runtime schema validation with Zod
      try {
        const validatedOutput = validateMlOutput(parsedJson);
        logger.info('ML prediction completed successfully', {
          latitude,
          longitude,
          year,
          predictedClass: validatedOutput.predicted_drought_class,
          severity: validatedOutput.severity_label,
          durationMs,
        });
        resolve({
          data: validatedOutput,
          durationMs,
        });
      } catch (valErr) {
        logger.error('ML output schema validation failed', {
          error: valErr.errors || valErr.message,
          durationMs,
        });
        reject(
          new MlServiceError(
            'ML output schema validation failure',
            'Prediction output format unrecognized.'
          )
        );
      }
    });
  });
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
