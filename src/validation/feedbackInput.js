/**
 * Operator Feedback Validation Schema (Zod)
 * =========================================
 * Enforces strict validation on incoming operator field reports.
 */

const { z } = require('zod');

const feedbackInputSchema = z.object({
  location_name: z.string().min(1).max(100).trim(),
  latitude: z.number().min(-90.0).max(90.0),
  longitude: z.number().min(-180.0).max(180.0),
  observed_year: z.number().int().min(2000).max(2100),
  observed_condition: z.enum(['normal_wet', 'moderate_stress', 'severe_drought']),
  borehole_yield_status: z.enum(['full_capacity', 'reduced_yield', 'dry_or_depleted']),
  water_table_depth_meters: z.number().min(0).max(1000).nullable().optional(),
  notes: z.string().max(500).trim().optional(),
  submitted_by: z.string().max(100).trim().optional(),
});

function safeValidateFeedbackInput(input) {
  return feedbackInputSchema.safeParse(input);
}

module.exports = {
  feedbackInputSchema,
  safeValidateFeedbackInput,
};
