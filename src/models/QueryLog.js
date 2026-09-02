/**
 * QueryLog Mongoose Model
 * =======================
 * Persists validated drought prediction results for audit and historical tracking.
 * Designed with strict field boundaries and safe indexing.
 */

const mongoose = require('mongoose');

const queryLogSchema = new mongoose.Schema(
  {
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
    year: {
      type: Number,
      required: true,
      min: 1700,
      max: 2100,
    },
    predicted_drought_class: {
      type: Number,
      required: true,
      enum: [0, 1, 2],
    },
    severity_label: {
      type: String,
      required: true,
      trim: true,
    },
    confidence_probabilities: {
      class_0: { type: Number, required: true, min: 0, max: 1 },
      class_1: { type: Number, required: true, min: 0, max: 1 },
      class_2: { type: Number, required: true, min: 0, max: 1 },
    },
    grid_cell: {
      requested_lat: Number,
      requested_lon: Number,
      selected_lat: Number,
      selected_lon: Number,
      distance_km: Number,
    },
    service_mode: {
      type: String,
      trim: true,
    },
    execution_duration_ms: {
      type: Number,
      min: 0,
    },
    timestamp: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    versionKey: false,
    timestamps: false,
  }
);

// Compound indexes for geographic and temporal query optimization
queryLogSchema.index({ timestamp: -1 });
queryLogSchema.index({ latitude: 1, longitude: 1 });
queryLogSchema.index({ predicted_drought_class: 1 });

const QueryLog = mongoose.models.QueryLog || mongoose.model('QueryLog', queryLogSchema);

module.exports = QueryLog;
