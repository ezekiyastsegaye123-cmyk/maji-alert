/**
 * Socket.io Handler for FRDASCR
 * =================================
 * Manages real-time WebSocket communication for drought predictions.
 * Enforces per-socket concurrency, cooldown throttling, and zero-broadcast privacy.
 */

const { safeValidatePredictionInput } = require('../validation/predictionInput');
const mlService = require('../services/mlService');
const QueryLog = require('../models/QueryLog');
const logger = require('../services/logger');

// Centralized Socket.io event name constants
const EVENTS = {
  PREDICT: 'drought:predict',
  STATUS: 'drought:status',
  RESULT: 'drought:prediction_result',
  ERROR: 'drought:prediction_error',
};

const MIN_COOLDOWN_MS = 2000;

/**
 * Attaches FRDASCR event listeners to Socket.io instance.
 *
 * @param {import('socket.io').Server} io
 */
function registerSocketHandlers(io) {
  io.on('connection', (socket) => {
    logger.debug('Client connected to Socket.io', { socketId: socket.id });

    // Track state per connection
    socket.inFlightPrediction = false;
    socket.lastRequestTimestamp = 0;

    socket.on(EVENTS.PREDICT, async (payload) => {
      const now = Date.now();

      // 1. Throttling and duplicate request check
      if (socket.inFlightPrediction) {
        return socket.emit(EVENTS.ERROR, {
          message: 'A prediction is already in progress for your session. Please wait.',
          code: 'IN_FLIGHT',
        });
      }

      if (now - socket.lastRequestTimestamp < MIN_COOLDOWN_MS) {
        return socket.emit(EVENTS.ERROR, {
          message: 'Please wait a moment before submitting another request.',
          code: 'RATE_LIMITED',
        });
      }

      socket.inFlightPrediction = true;
      socket.lastRequestTimestamp = now;

      // 2. Strict Zod input validation
      const validationResult = safeValidatePredictionInput(payload);
      if (!validationResult.success) {
        socket.inFlightPrediction = false;
        const rawIssues = validationResult.error.issues || validationResult.error.errors || [];
        const formattedErrors = rawIssues.map((e) => ({
          field: Array.isArray(e.path) ? e.path.join('.') : '',
          message: e.message,
        }));
        logger.warn('Socket prediction input validation rejected', {
          socketId: socket.id,
          errors: formattedErrors,
        });
        return socket.emit(EVENTS.ERROR, {
          message: 'Validation failed: Invalid coordinate or parameter inputs.',
          errors: formattedErrors,
          code: 'VALIDATION_ERROR',
        });
      }

      const { latitude, longitude, year } = validationResult.data;

      try {
        // 3. Status notification to requesting socket only
        socket.emit(EVENTS.STATUS, {
          step: 'processing',
          message: 'Executing climate model and solar-cycle analysis...',
        });

        // 4. Invoke Python ML service
        const prediction = await mlService.executePrediction({ latitude, longitude, year });

        // 5. Asynchronous persistence to MongoDB (non-blocking for client response)
        try {
          if (QueryLog.db && QueryLog.db.readyState === 1) {
            await QueryLog.create({
              latitude,
              longitude,
              year,
              predicted_drought_class: prediction.data.predicted_drought_class,
              severity_label: prediction.data.severity_label,
              confidence_probabilities: prediction.data.confidence_probabilities,
              grid_cell: prediction.data.grid_cell,
              service_mode: prediction.data.service_mode,
              execution_duration_ms: prediction.durationMs,
            });
            logger.debug('Query log persisted to MongoDB', { socketId: socket.id });
          } else {
            logger.debug('Database not connected; skipping QueryLog persistence');
          }
        } catch (dbErr) {
          logger.warn('Non-fatal MongoDB query log error', { error: dbErr.message });
        }

        // 6. Emit result STRICTLY to requesting socket (zero-broadcast privacy)
        socket.emit(EVENTS.RESULT, {
          ...prediction.data,
          execution_duration_ms: prediction.durationMs,
        });
      } catch (err) {
        logger.error('Socket prediction execution failed', {
          socketId: socket.id,
          error: err.message,
        });
        const clientMsg =
          err instanceof mlService.MlServiceError
            ? err.clientMessage
            : 'Prediction service temporarily unavailable.';
        socket.emit(EVENTS.ERROR, {
          message: clientMsg,
          code: err.statusCode || 500,
        });
      } finally {
        socket.inFlightPrediction = false;
      }
    });

    socket.on('disconnect', () => {
      logger.debug('Client disconnected from Socket.io', { socketId: socket.id });
      socket.inFlightPrediction = false;
    });
  });
}

module.exports = {
  registerSocketHandlers,
  EVENTS,
};
