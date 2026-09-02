/**
 * Unit Tests: Prediction Input Validation (Zod)
 * ==============================================
 */

const {
  validatePredictionInput,
  safeValidatePredictionInput,
} = require('../../src/validation/predictionInput');

describe('Prediction Input Validation', () => {
  describe('Valid Coordinates', () => {
    it('accepts valid coordinates within Ethiopian / global boundaries', () => {
      const input = { latitude: 4.88, longitude: 38.08, year: 2026 };
      const parsed = validatePredictionInput(input);
      expect(parsed.latitude).toBe(4.88);
      expect(parsed.longitude).toBe(38.08);
      expect(parsed.year).toBe(2026);
    });

    it('defaults year to 2026 if omitted', () => {
      const input = { latitude: 9.63, longitude: 39.53 };
      const parsed = validatePredictionInput(input);
      expect(parsed.latitude).toBe(9.63);
      expect(parsed.longitude).toBe(39.53);
      expect(parsed.year).toBe(2026);
    });

    it('accepts extreme boundary values (-90, 90, -180, 180)', () => {
      expect(() =>
        validatePredictionInput({ latitude: -90, longitude: -180, year: 2000 })
      ).not.toThrow();
      expect(() =>
        validatePredictionInput({ latitude: 90, longitude: 180, year: 2000 })
      ).not.toThrow();
    });
  });

  describe('Invalid Latitude & Longitude Bounds', () => {
    it('rejects latitude below -90.0', () => {
      const result = safeValidatePredictionInput({ latitude: -90.01, longitude: 38.0 });
      expect(result.success).toBe(false);
      expect(result.error.errors[0].message).toContain('>= -90.0');
    });

    it('rejects latitude above 90.0', () => {
      const result = safeValidatePredictionInput({ latitude: 90.01, longitude: 38.0 });
      expect(result.success).toBe(false);
      expect(result.error.errors[0].message).toContain('<= 90.0');
    });

    it('rejects longitude below -180.0', () => {
      const result = safeValidatePredictionInput({ latitude: 4.88, longitude: -180.1 });
      expect(result.success).toBe(false);
      expect(result.error.errors[0].message).toContain('>= -180.0');
    });

    it('rejects longitude above 180.0', () => {
      const result = safeValidatePredictionInput({ latitude: 4.88, longitude: 180.1 });
      expect(result.success).toBe(false);
      expect(result.error.errors[0].message).toContain('<= 180.0');
    });
  });

  describe('Non-Finite, Type, and Missing Value Rejections', () => {
    it('rejects NaN for latitude or longitude', () => {
      expect(safeValidatePredictionInput({ latitude: NaN, longitude: 38.0 }).success).toBe(false);
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: NaN }).success).toBe(false);
    });

    it('rejects Infinity and -Infinity', () => {
      expect(safeValidatePredictionInput({ latitude: Infinity, longitude: 38.0 }).success).toBe(false);
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: -Infinity }).success).toBe(false);
    });

    it('rejects string representations of numbers', () => {
      expect(safeValidatePredictionInput({ latitude: '4.88', longitude: 38.08 }).success).toBe(false);
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: '38.08' }).success).toBe(false);
    });

    it('rejects missing latitude or longitude', () => {
      expect(safeValidatePredictionInput({ longitude: 38.08 }).success).toBe(false);
      expect(safeValidatePredictionInput({ latitude: 4.88 }).success).toBe(false);
      expect(safeValidatePredictionInput({}).success).toBe(false);
    });

    it('rejects years outside [1700, 2100]', () => {
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: 38.08, year: 1699 }).success).toBe(false);
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: 38.08, year: 2101 }).success).toBe(false);
    });

    it('rejects non-integer years', () => {
      expect(safeValidatePredictionInput({ latitude: 4.88, longitude: 38.08, year: 2026.5 }).success).toBe(false);
    });
  });

  describe('Strict Object Integrity (No Unexpected Fields)', () => {
    it('rejects payloads containing extra or unapproved fields', () => {
      const result = safeValidatePredictionInput({
        latitude: 4.88,
        longitude: 38.08,
        year: 2026,
        maliciousField: 'exploit',
        admin: true,
      });
      expect(result.success).toBe(false);
      expect(result.error.errors[0].message).toMatch(/unrecognized|unexpected/i);
    });
  });
});
