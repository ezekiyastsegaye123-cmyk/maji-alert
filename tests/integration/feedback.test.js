/**
 * Integration Tests: Operator Feedback API Endpoints
 * ==================================================
 */

const request = require('supertest');
const { app } = require('../../server');

describe('Operator Feedback API (/api/feedback)', () => {
  const validFeedback = {
    location_name: 'Dubuluk Deep Borehole',
    latitude: 4.88,
    longitude: 38.08,
    observed_year: 2026,
    observed_condition: 'moderate_stress',
    borehole_yield_status: 'reduced_yield',
    water_table_depth_meters: 85.5,
    notes: 'Static water level dropped 3.2m following delayed Genna rains.',
    submitted_by: 'Borehole Technician Gemechu',
  };

  it('successfully accepts valid operator feedback', async () => {
    const res = await request(app)
      .post('/api/feedback')
      .send(validFeedback);

    expect([200, 201]).toContain(res.status);
    expect(res.body).toHaveProperty('status');
    expect(['success', 'accepted_ephemeral']).toContain(res.body.status);
  });

  it('rejects feedback with missing required fields', async () => {
    const res = await request(app)
      .post('/api/feedback')
      .send({
        latitude: 4.88,
        longitude: 38.08,
      });

    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty('error');
    expect(res.body.error).toContain('Invalid operator feedback');
  });

  it('rejects invalid condition enum values', async () => {
    const res = await request(app)
      .post('/api/feedback')
      .send({
        ...validFeedback,
        observed_condition: 'catastrophic_flood',
      });

    expect(res.status).toBe(400);
    expect(res.body.details).toBeDefined();
  });

  it('rejects out-of-bounds coordinates', async () => {
    const res = await request(app)
      .post('/api/feedback')
      .send({
        ...validFeedback,
        latitude: 95.0,
      });

    expect(res.status).toBe(400);
  });

  it('retrieves recent feedback list', async () => {
    const res = await request(app).get('/api/feedback');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('status', 'ok');
    expect(res.body).toHaveProperty('data');
    expect(Array.isArray(res.body.data)).toBe(true);
  });
});
