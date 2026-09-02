/**
 * Unit Tests: Python ML Subprocess Safeguards & Error Recovery
 * ============================================================
 */

const { EventEmitter } = require('events');
const childProcess = require('child_process');
const { executePrediction, MlServiceError, getActiveJobsCount } = require('../../src/services/mlService');

describe('ML Subprocess Safeguards', () => {
  let spawnSpy;

  afterEach(() => {
    if (spawnSpy) {
      spawnSpy.mockRestore();
    }
  });

  it('handles non-zero exit codes safely without crashing', async () => {
    const mockChild = new EventEmitter();
    mockChild.stdout = new EventEmitter();
    mockChild.stderr = new EventEmitter();
    mockChild.kill = jest.fn();

    spawnSpy = jest.spyOn(childProcess, 'spawn').mockImplementation(() => {
      process.nextTick(() => {
        mockChild.stderr.emit('data', 'Traceback (most recent call last): RuntimeError: Model corrupt\n');
        mockChild.emit('close', 1, null);
      });
      return mockChild;
    });

    await expect(
      executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 })
    ).rejects.toThrow(MlServiceError);
  });

  it('masks raw Python errors and presents a safe client message', async () => {
    const mockChild = new EventEmitter();
    mockChild.stdout = new EventEmitter();
    mockChild.stderr = new EventEmitter();
    mockChild.kill = jest.fn();

    spawnSpy = jest.spyOn(childProcess, 'spawn').mockImplementation(() => {
      process.nextTick(() => {
        mockChild.stderr.emit('data', 'Internal database failure password=secret123\n');
        mockChild.emit('close', 2, null);
      });
      return mockChild;
    });

    try {
      await executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 });
      throw new Error('Expected executePrediction to throw, but it succeeded');
    } catch (err) {
      expect(err).toBeInstanceOf(MlServiceError);
      expect(err.clientMessage).toBe('Prediction service temporarily unavailable.');
      expect(err.clientMessage).not.toContain('password');
    }
  });

  it('safely rejects malformed non-JSON stdout', async () => {
    const mockChild = new EventEmitter();
    mockChild.stdout = new EventEmitter();
    mockChild.stderr = new EventEmitter();
    mockChild.kill = jest.fn();

    spawnSpy = jest.spyOn(childProcess, 'spawn').mockImplementation(() => {
      process.nextTick(() => {
        mockChild.stdout.emit('data', '<html><head><title>500 Error</title></head><body>Crash</body></html>');
        mockChild.emit('close', 0, null);
      });
      return mockChild;
    });

    await expect(
      executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 })
    ).rejects.toThrow('Malformed JSON');
  });

  it('safely rejects empty stdout from Python subprocess', async () => {
    const mockChild = new EventEmitter();
    mockChild.stdout = new EventEmitter();
    mockChild.stderr = new EventEmitter();
    mockChild.kill = jest.fn();

    spawnSpy = jest.spyOn(childProcess, 'spawn').mockImplementation(() => {
      process.nextTick(() => {
        mockChild.emit('close', 0, null);
      });
      return mockChild;
    });

    await expect(
      executePrediction({ latitude: 4.88, longitude: 38.08, year: 2026 })
    ).rejects.toThrow('Empty output');
  });
});
