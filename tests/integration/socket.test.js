/**
 * Integration Tests: Socket.io Real-Time Protocol & Isolation
 * =============================================================
 */

const { createServer } = require('http');
const { Server } = require('socket.io');
const Client = require('socket.io-client');
const { registerSocketHandlers, EVENTS } = require('../../src/socket/socketHandler');
const mlService = require('../../src/services/mlService');

describe('Socket.io Real-Time Integration & Privacy Isolation', () => {
  let io, server, port;
  let clientSocket1, clientSocket2;

  const mockPredictionData = {
    data: {
      predicted_drought_class: 1,
      severity_label: 'Moderate Drought',
      confidence_probabilities: { class_0: 0.3, class_1: 0.5, class_2: 0.2 },
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
    durationMs: 820,
  };

  beforeAll((done) => {
    server = createServer();
    io = new Server(server);
    registerSocketHandlers(io);

    server.listen(() => {
      port = server.address().port;
      done();
    });
  });

  afterAll(() => {
    io.close();
    server.close();
  });

  beforeEach((done) => {
    clientSocket1 = new Client(`http://localhost:${port}`, { reconnection: false });
    clientSocket2 = new Client(`http://localhost:${port}`, { reconnection: false });

    let connected = 0;
    const checkDone = () => {
      connected += 1;
      if (connected === 2) done();
    };

    clientSocket1.on('connect', checkDone);
    clientSocket2.on('connect', checkDone);
  });

  afterEach(() => {
    if (clientSocket1 && clientSocket1.connected) clientSocket1.disconnect();
    if (clientSocket2 && clientSocket2.connected) clientSocket2.disconnect();
  });

  it('delivers prediction result to requesting socket', (done) => {
    const spy = jest.spyOn(mlService, 'executePrediction').mockResolvedValue(mockPredictionData);

    clientSocket1.emit(EVENTS.PREDICT, {
      latitude: 4.88,
      longitude: 38.08,
      year: 2026,
    });

    clientSocket1.on(EVENTS.RESULT, (data) => {
      expect(data.predicted_drought_class).toBe(1);
      expect(data.severity_label).toBe('Moderate Drought');
      expect(data.execution_duration_ms).toBe(820);
      spy.mockRestore();
      done();
    });
  });

  it('ensures zero-broadcast privacy (Client 2 does NOT receive Client 1 result)', (done) => {
    const spy = jest.spyOn(mlService, 'executePrediction').mockResolvedValue(mockPredictionData);

    let client2Received = false;

    clientSocket2.on(EVENTS.RESULT, () => {
      client2Received = true;
    });

    clientSocket1.emit(EVENTS.PREDICT, {
      latitude: 4.88,
      longitude: 38.08,
      year: 2026,
    });

    clientSocket1.on(EVENTS.RESULT, () => {
      // Allow brief tick to ensure socket2 didn't receive event
      setTimeout(() => {
        expect(client2Received).toBe(false);
        spy.mockRestore();
        done();
      }, 50);
    });
  });

  it('rejects invalid coordinates with drought:prediction_error', (done) => {
    clientSocket1.emit(EVENTS.PREDICT, {
      latitude: 999.0, // Invalid latitude
      longitude: 38.08,
      year: 2026,
    });

    clientSocket1.on(EVENTS.ERROR, (err) => {
      expect(err.code).toBe('VALIDATION_ERROR');
      expect(err.errors).toBeDefined();
      done();
    });
  });

  it('rejects concurrent in-flight requests on the same socket', (done) => {
    let callCount = 0;
    jest.spyOn(mlService, 'executePrediction').mockImplementation(() => {
      callCount += 1;
      return new Promise((resolve) => setTimeout(() => resolve(mockPredictionData), 100));
    });

    // Send first request
    clientSocket1.emit(EVENTS.PREDICT, { latitude: 4.88, longitude: 38.08, year: 2026 });

    // Send duplicate immediately while first is in flight
    clientSocket1.emit(EVENTS.PREDICT, { latitude: 4.88, longitude: 38.08, year: 2026 });

    clientSocket1.on(EVENTS.ERROR, (err) => {
      expect(err.code).toBe('IN_FLIGHT');
      expect(err.message).toContain('already in progress');
      setTimeout(() => done(), 150);
    });
  });
});
