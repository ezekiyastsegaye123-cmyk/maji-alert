/**
 * OperatorFeedback Mongoose Model
 * ===============================
 * Records ground-truth observations and borehole feedback from water point operators
 * across Borana Zone (e.g. static water level, pump operational status, herd pressure).
 */

const mongoose = require('mongoose');

const operatorFeedbackSchema = new mongoose.Schema(
  {
    location_name: {
      type: String,
      required: true,
      trim: true,
      maxlength: 100,
    },
    latitude: {
      type: Number,
      required: true,
      min: -90,
      max: 90,
    },
    longitude: {
      type: Number,
      required: true,
      min: -180,
      max: 180,
    },
    observed_year: {
      type: Number,
      required: true,
      min: 2000,
      max: 2100,
    },
    observed_condition: {
      type: String,
      required: true,
      enum: ['normal_wet', 'moderate_stress', 'severe_drought'],
    },
    borehole_yield_status: {
      type: String,
      required: true,
      enum: ['full_capacity', 'reduced_yield', 'dry_or_depleted'],
    },
    water_table_depth_meters: {
      type: Number,
      min: 0,
      max: 1000,
      default: null,
    },
    notes: {
      type: String,
      trim: true,
      maxlength: 500,
      default: '',
    },
    submitted_by: {
      type: String,
      trim: true,
      maxlength: 100,
      default: 'Anonymous Operator',
    },
    timestamp: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: true,
  }
);

operatorFeedbackSchema.index({ latitude: 1, longitude: 1, observed_year: -1 });

module.exports = mongoose.model('OperatorFeedback', operatorFeedbackSchema);
