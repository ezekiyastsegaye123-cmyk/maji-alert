/**
 * Maji Alert — Production Express + Socket.io Server
 * ==================================================
 * Serving real-time drought notifications for the Borana Zone, Ethiopia.
 */

const http = require('http');
const path = require('path');
const express = require('express');
const { Server } = require('socket.io');
const mongoose = require('mongoose');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

const config = require('./src/config/env');
const logger = require('./src/services/logger');
const { registerSocketHandlers } = require('./src/socket/socketHandler');
const { safeValidatePredictionInput } = require('./src/validation/predictionInput');
const mlService = require('./src/services/mlService');
const QueryLog = require('./src/models/QueryLog');

const app = express();
const server = http.createServer(app);

// 1. Security Headers (Helmet)
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        connectSrc: ["'self'", 'ws:', 'wss:'],
        imgSrc: ["'self'", 'data:'],
        fontSrc: ["'self'"],
      },
    },
    crossOriginEmbedderPolicy: false,
  })
);

// 2. Strict CORS Configuration
const corsOptions = {
  origin: config.CORS_ORIGIN === '*' ? true : config.CORS_ORIGIN.split(','),
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
};
app.use(cors(corsOptions));

// 3. Request Body Parsing with Strict Size Limit
app.use(express.json({ limit: '10kb' }));

// 4. Rate Limiting for API routes
const apiLimiter = rateLimit({
  windowMs: config.RATE_LIMIT_WINDOW_MS,
  max: config.RATE_LIMIT_MAX_REQUESTS,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'Too many requests from this IP. Please try again later.',
    status: 429,
  },
});
app.use('/api/', apiLimiter);

// 5. Static Assets (Low-bandwidth frontend with cache control)
app.get('/favicon.ico', (req, res) => res.status(204).end());

app.use(
  express.static(path.join(__dirname, 'public'), {
    etag: false,
    maxAge: 0,
    setHeaders: (res) => {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
    },
  })
);

// 6. Health Endpoint (Zero Secret Leakage)
app.get('/health', async (req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbStatusMap = {
    0: 'disconnected',
    1: 'connected',
    2: 'connecting',
    3: 'disconnecting',
  };

  let mlEngineStatus = 'unknown';
  let timeout;
  try {
    const controller = new AbortController();
    timeout = setTimeout(() => controller.abort(), 1500);
    const mlRes = await fetch(`${config.ML_SERVICE_URL}/ready`, { signal: controller.signal });
    mlEngineStatus = mlRes.ok ? 'ready' : 'degraded';
  } catch (_) {
    mlEngineStatus = 'offline';
  } finally {
    if (timeout) clearTimeout(timeout);
  }

  const healthData = {
    status: 'ok',
    service: 'Maji Alert API',
    uptime_seconds: Math.floor(process.uptime()),
    timestamp: new Date().toISOString(),
    components: {
      database: dbStatusMap[dbState] || 'unknown',
      ml_engine: mlEngineStatus,
    },
  };

  const isHealthy = dbState === 1 || dbState === 0; // DB down is non-fatal for core service health
  res.status(isHealthy ? 200 : 503).json(healthData);
});

// 7. REST Prediction Endpoint (Alternative to Socket.io for API clients)
app.post('/api/predict', async (req, res) => {
  const validationResult = safeValidatePredictionInput(req.body);
  if (!validationResult.success) {
    const rawIssues = validationResult.error.issues || validationResult.error.errors || [];
    return res.status(400).json({
      error: 'Invalid coordinate or parameter inputs',
      details: rawIssues.map((e) => ({
        field: Array.isArray(e.path) ? e.path.join('.') : '',
        message: e.message,
      })),
      status: 400,
    });
  }

  const { latitude, longitude, year } = validationResult.data;

  try {
    const prediction = await mlService.executePrediction({ latitude, longitude, year });

    // Asynchronous audit logging
    if (mongoose.connection.readyState === 1) {
      QueryLog.create({
        latitude,
        longitude,
        year,
        predicted_drought_class: prediction.data.predicted_drought_class,
        severity_label: prediction.data.severity_label,
        confidence_probabilities: prediction.data.confidence_probabilities,
        grid_cell: prediction.data.grid_cell,
        service_mode: prediction.data.service_mode,
        execution_duration_ms: prediction.durationMs,
      }).catch((err) => logger.warn('Non-fatal MongoDB query log error', { error: err.message }));
    }

    return res.status(200).json({
      ...prediction.data,
      execution_duration_ms: prediction.durationMs,
    });
  } catch (err) {
    logger.error('REST prediction execution failed', { error: err.message });
    const statusCode = err instanceof mlService.MlServiceError ? err.statusCode : 500;
    const clientMessage =
      err instanceof mlService.MlServiceError
        ? err.clientMessage
        : 'Prediction service temporarily unavailable.';

    return res.status(statusCode).json({
      error: clientMessage,
      status: statusCode,
    });
  }
});

// 8. Attach Socket.io with Matching CORS and 60s pingTimeout
const io = new Server(server, {
  cors: corsOptions,
  pingTimeout: 60000,
  pingInterval: 25000,
});

registerSocketHandlers(io);

// 9. Database Connection Initialization (Non-blocking / Graceful)
async function connectDatabase() {
  if (config.NODE_ENV === 'test') return;

  try {
    logger.info('Connecting to MongoDB...', { uri: config.MONGODB_URI.replace(/\/\/.*@/, '//***@') });
    await mongoose.connect(config.MONGODB_URI, {
      serverSelectionTimeoutMS: 5000,
    });
    logger.info('MongoDB connected successfully');
  } catch (err) {
    logger.warn('Initial MongoDB connection failed. Operating in degraded mode (no persistent logs)', {
      error: err.message,
    });
  }
}

// 10. Graceful Shutdown
let isShuttingDown = false;

async function handleGracefulShutdown(signal) {
  if (isShuttingDown) return;
  isShuttingDown = true;
  logger.info(`Received ${signal}. Initiating graceful shutdown...`);

  // Stop accepting new HTTP connections
  server.close(async () => {
    logger.info('HTTP server closed');

    // Close Socket.io connections
    try {
      io.close();
      logger.info('Socket.io server closed');
    } catch (ioErr) {
      logger.error('Error closing Socket.io', { error: ioErr.message });
    }

    // Close MongoDB connection
    try {
      if (mongoose.connection.readyState !== 0) {
        await mongoose.connection.close(false);
        logger.info('MongoDB connection closed');
      }
    } catch (dbErr) {
      logger.error('Error closing MongoDB connection', { error: dbErr.message });
    }

    logger.info('Graceful shutdown completed successfully');
    process.exit(0);
  });

  // Force exit after 10s if connections fail to close
  setTimeout(() => {
    logger.error('Graceful shutdown timed out. Forcing process exit.');
    process.exit(1);
  }, 10000).unref();
}

process.on('SIGTERM', () => handleGracefulShutdown('SIGTERM'));
process.on('SIGINT', () => handleGracefulShutdown('SIGINT'));

// 11. Server Start (only if not required by a test harness)
if (require.main === module) {
  connectDatabase().then(() => {
    server.listen(config.PORT, () => {
      logger.info(`Maji Alert server running on port ${config.PORT} [${config.NODE_ENV}]`);
    });
  });
}

module.exports = { app, server, io };
