/**
 * Unit Tests: Python ML HTTP Client & Error Recovery
 * ==================================================
 * Validates HTTP communication, timeout handling, retry logic,
 * and error masking with persistent FastAPI ML microservice.
 */

const { executePrediction, MlServiceError, getActiveJobsCount } = require('../../src/services/mlService');

describe('ML HTTP Client Safeguards', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.restoreAllMocks();
  });

  it('successfully fetches and validates prediction from FastAPI', async () => {
    const mockApiResponse = {
      predicted_drought_class: 2,
      severity_label: 'Severe Drought',
      confidence_probabilities: {
        class_0: 0.2605,
        class_1: 0.2364,
        class_2: 0.5031,
      },
      grid_cell: {
        requested_lat: 4.88,
        requested_lon: 38.08,
        selected_lat: 4.75,
        selected_lon: 38.25,
        distance_km: 23.74,
      },
      year: 2026,
      service_mode: 'prospective_solar_projection',
    };

    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockApiResponse,
    });

    const result = await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
    expect(result.data.predicted_drought_class).toBe(2);
    expect(result.data.severity_label).toBe('Severe Drought');
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('handles FastAPI 400/422 validation errors with status 400', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'Latitude out of bounds' }),
    });

    try {
      await executePrediction({ latitude: 95.0, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.statusCode).toBe(400);
      expect(err.clientMessage).toBe('Invalid coordinate or parameter inputs.');
    }
  });

  it('handles FastAPI 503 service unavailable cleanly', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: { ready: false, error: 'Model initializing' } }),
    });

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.statusCode).toBe(503);
      expect(err.clientMessage).toContain('temporarily unavailable');
    }
  });

  it('handles request timeout (AbortError) with status 504', async () => {
    global.fetch = jest.fn().mockImplementation(() => {
      const abortErr = new Error('The operation was aborted');
      abortErr.name = 'AbortError';
      return Promise.reject(abortErr);
    });

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.statusCode).toBe(504);
      expect(err.clientMessage).toBe('Prediction request timed out. Please try again.');
    }
  });

  it('handles malformed non-JSON response from FastAPI', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError('Unexpected token < in JSON at position 0');
      },
    });

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.statusCode).toBe(502);
      expect(err.clientMessage).toBe('Prediction output format unrecognized.');
    }
  });

  it('handles connection refused (ECONNREFUSED) with retry and returns 503', async () => {
    const connErr = new Error('fetch failed');
    connErr.cause = { code: 'ECONNREFUSED' };

    global.fetch = jest.fn().mockRejectedValue(connErr);

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.statusCode).toBe(503);
      expect(err.clientMessage).toContain('currently unavailable');
      // Verify that transient retry was attempted (maxAttempts = 2)
      expect(global.fetch).toHaveBeenCalledTimes(2);
    }
  });

  it('masks internal secrets from client errors', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ secret: 'mongodb_key_12345' }),
    });

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.clientMessage).not.toContain('mongodb_key');
      expect(err.clientMessage).toBe('Prediction service encountered an internal error.');
    }
  });
});
