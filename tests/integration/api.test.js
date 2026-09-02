/**
 * Integration Tests: Express API Endpoints & Health Check
 * ========================================================
 */

const request = require('supertest');
const { app } = require('../../server');
const mlService = require('../../src/services/mlService');

describe('REST API & Server Endpoints', () => {
  describe('GET /health', () => {
    it('returns status 200 and operational health metadata without exposing secrets', async () => {
      const res = await request(app).get('/health');

      expect(res.status).toBe(200);
      expect(res.body.status).toBe('ok');
      expect(res.body.service).toBe('Maji Alert API');
      expect(res.body).toHaveProperty('uptime_seconds');
      expect(res.body).toHaveProperty('components');

      // Security check: Verify no secret leakage
      const rawBody = JSON.stringify(res.body);
      expect(rawBody).not.toContain('mongodb://');
      expect(rawBody).not.toContain('password');
      expect(rawBody).not.toContain('/home/');
    });
  });

  describe('GET / (Static Asset Serving)', () => {
    it('serves the low-bandwidth HTML entrypoint', async () => {
      const res = await request(app).get('/');
      expect(res.status).toBe(200);
      expect(res.headers['content-type']).toContain('text/html');
      expect(res.text).toContain('Maji Alert');
      expect(res.text).toContain('Borana');
    });

    it('serves CSS stylesheet and JS bundle', async () => {
      const cssRes = await request(app).get('/styles.css');
      expect(cssRes.status).toBe(200);
      expect(cssRes.headers['content-type']).toContain('text/css');

      const jsRes = await request(app).get('/app.js');
      expect(jsRes.status).toBe(200);
      expect(jsRes.headers['content-type']).toContain('javascript');
    });
  });

  describe('POST /api/predict', () => {
    it('returns 400 on invalid latitude or longitude bounds', async () => {
      const res = await request(app)
        .post('/api/predict')
        .send({ latitude: 95.0, longitude: 38.08 });

      expect(res.status).toBe(400);
      expect(res.body.error).toContain('Invalid coordinate');
      expect(res.body.details).toBeDefined();
    });

    it('returns 400 on unexpected prohibited properties', async () => {
      const res = await request(app)
        .post('/api/predict')
        .send({ latitude: 4.88, longitude: 38.08, inject: true });

      expect(res.status).toBe(400);
      expect(res.body.error).toContain('Invalid coordinate');
    });

    it('executes valid prediction request and returns calibrated result', async () => {
      // Mock executePrediction to ensure fast, deterministic unit response
      const mockResult = {
        data: {
          predicted_drought_class: 2,
          severity_label: 'Severe Drought',
          confidence_probabilities: { class_0: 0.2, class_1: 0.3, class_2: 0.5 },
          grid_cell: {
            requested_lat: 4.88,
            requested_lon: 38.08,
            selected_lat: 4.75,
            selected_lon: 38.25,
            distance_km: 23.74,
          },
          year: 2026,
          service_mode: 'prospective_solar_projection',
        },
        durationMs: 1450,
      };

      const spy = jest.spyOn(mlService, 'executePrediction').mockResolvedValue(mockResult);

      const res = await request(app)
        .post('/api/predict')
        .send({ latitude: 4.88, longitude: 38.08, year: 2026 });

      expect(res.status).toBe(200);
      expect(res.body.predicted_drought_class).toBe(2);
      expect(res.body.severity_label).toBe('Severe Drought');
      expect(res.body.execution_duration_ms).toBe(1450);

      spy.mockRestore();
    });
  });
});
