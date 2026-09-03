/**
 * Unit Tests: Python ML Output Schema Validation
 * ===============================================
 */

const { validateMlOutput, safeValidateMlOutput } = require('../../src/validation/mlOutput');

describe('ML Output Validation', () => {
  const validOutput = {
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

  it('validates a correct predict_service output payload', () => {
    const parsed = validateMlOutput(validOutput);
    expect(parsed.predicted_drought_class).toBe(2);
    expect(parsed.severity_label).toBe('Severe Drought');
    expect(parsed.confidence_probabilities.class_2).toBe(0.5031);
  });

  it('accepts classes 0, 1, and 2', () => {
    [0, 1, 2].forEach((cls) => {
      const payload = { ...validOutput, predicted_drought_class: cls };
      expect(safeValidateMlOutput(payload).success).toBe(true);
    });
  });

  it('rejects unapproved class values (e.g. 3, -1, null)', () => {
    expect(safeValidateMlOutput({ ...validOutput, predicted_drought_class: 3 }).success).toBe(false);
    expect(safeValidateMlOutput({ ...validOutput, predicted_drought_class: -1 }).success).toBe(false);
    expect(safeValidateMlOutput({ ...validOutput, predicted_drought_class: 'severe' }).success).toBe(false);
  });

  it('rejects probability distributions that do not sum to ~1.0', () => {
    const invalidProbabilities = {
      ...validOutput,
      confidence_probabilities: {
        class_0: 0.8,
        class_1: 0.8,
        class_2: 0.8,
      },
    };
    expect(safeValidateMlOutput(invalidProbabilities).success).toBe(false);
  });

  it('rejects negative probability values', () => {
    const negativeProbabilities = {
      ...validOutput,
      confidence_probabilities: {
        class_0: -0.1,
        class_1: 0.6,
        class_2: 0.5,
      },
    };
    expect(safeValidateMlOutput(negativeProbabilities).success).toBe(false);
  });

  it('rejects missing grid_cell details', () => {
    const noGrid = { ...validOutput };
    delete noGrid.grid_cell;
    expect(safeValidateMlOutput(noGrid).success).toBe(false);
  });

  it('rejects missing severity label or year', () => {
    const noYear = { ...validOutput };
    delete noYear.year;
    expect(safeValidateMlOutput(noYear).success).toBe(false);
  });

  it('accepts valid model_confidence attribute', () => {
    const withConf = { ...validOutput, model_confidence: 0.5031 };
    const parsed = validateMlOutput(withConf);
    expect(parsed.model_confidence).toBe(0.5031);
  });
});
