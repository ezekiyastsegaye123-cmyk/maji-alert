/**
 * ML Output Validation Schema (Zod)
 * =================================
 * Enforces runtime schema integrity on stdout produced by predict_service.py.
 * Verifies drought class, confidence probabilities, grid cell info, and bounds.
 */

const { z } = require('zod');

const confidenceProbabilitiesSchema = z.object({
  class_0: z.number().min(0.0).max(1.0),
  class_1: z.number().min(0.0).max(1.0),
  class_2: z.number().min(0.0).max(1.0),
}).refine(
  (probs) => {
    const sum = probs.class_0 + probs.class_1 + probs.class_2;
    return sum >= 0.95 && sum <= 1.05;
  },
  { message: 'Confidence probabilities must sum to approximately 1.0' }
);

const gridCellSchema = z.object({
  requested_lat: z.number().min(-90).max(90),
  requested_lon: z.number().min(-180).max(180),
  selected_lat: z.number().min(-90).max(90),
  selected_lon: z.number().min(-180).max(180),
  distance_km: z.number().min(0),
});

const mlOutputSchema = z.object({
  predicted_drought_class: z.union([
    z.literal(0),
    z.literal(1),
    z.literal(2),
  ]),
  severity_label: z.string().min(1),
  confidence_probabilities: confidenceProbabilitiesSchema,
  grid_cell: gridCellSchema,
  year: z.number().int(),
  service_mode: z.string(),
});

function validateMlOutput(output) {
  try {
    return mlOutputSchema.parse(output);
  } catch (err) {
    if (err && !err.errors && err.issues) {
      err.errors = err.issues;
    }
    throw err;
  }
}

function safeValidateMlOutput(output) {
  const result = mlOutputSchema.safeParse(output);
  if (!result.success && result.error && !result.error.errors && result.error.issues) {
    result.error.errors = result.error.issues;
  }
  return result;
}

module.exports = {
  mlOutputSchema,
  validateMlOutput,
  safeValidateMlOutput,
};
