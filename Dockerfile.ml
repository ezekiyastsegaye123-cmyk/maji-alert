# ==============================================================================
# FRADSCR — Machine Learning Microservice (FastAPI + In-Memory Model Engine)
# ==============================================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install minimal OS dependencies required for C-extensions (netCDF4/HDF5/libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first for optimal Docker layer caching
COPY requirements-docker.txt .

# Install Python production dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application source code and scientific assets
COPY treering/ ./treering/
COPY models/ ./models/
COPY africa/ ./africa/
COPY data/ ./data/
COPY SN_y_tot_V2.0.csv .
COPY predict_service.py .

# Create non-root system user for runtime security hardening
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check using FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run FastAPI service with single Uvicorn worker to conserve RAM
CMD ["python", "predict_service.py", "--serve", "--port", "8000", "--host", "0.0.0.0"]
