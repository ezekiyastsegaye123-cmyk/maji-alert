# ==============================================================================
# FRADSCR — Unified Production Container (Node.js Express + Python FastAPI ML)
# Optimized for Cloud Deployment: Render, Railway, Fly.io, Hugging Face Spaces
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Node.js Production Dependencies
# ------------------------------------------------------------------------------
FROM node:20-alpine AS node-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

# ------------------------------------------------------------------------------
# Stage 2: Final Multi-Runtime Production Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Prevent Python buffering and bytecode write
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production \
    PORT=3000 \
    ML_SERVICE_URL=http://127.0.0.1:8000 \
    ML_REQUEST_TIMEOUT_MS=15000 \
    CALIBRATION_TEMPERATURE=0.35 \
    MONGODB_URI=mongodb://127.0.0.1:27017/fradscr

# Install Node.js 20, Supervisor, and C-dependencies for NetCDF / HDF5 / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    supervisor \
    libgomp1 \
    libhdf5-dev \
    libnetcdf-dev \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy Node production node_modules from builder
COPY --from=node-builder /app/node_modules ./node_modules
COPY package*.json ./

# Copy Application Code & Scientific Assets
COPY src/ ./src/
COPY public/ ./public/
COPY server.js .
COPY treering/ ./treering/
COPY models/ ./models/
COPY africa/ ./africa/
COPY data/ ./data/
COPY SN_y_tot_V2.0.csv .
COPY predict_service.py .

# Copy Supervisor configuration and startup entrypoint
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Health check verifies web gateway health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 3000

ENTRYPOINT ["/app/entrypoint.sh"]
