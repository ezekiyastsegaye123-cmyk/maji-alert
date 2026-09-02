# AGY — Production Phase 4 Scientific Validation, Geographic Holdout & Prediction Service

## ROLE

Act as a **senior machine-learning engineer, research-software engineer, scientific data engineer, and ML validation/QA engineer**.

You are responsible for completing **Phase 4 and the final evaluation deliverables** for the EGATE Heliophysics research project:

> **"Using Tree Rings to Study Solar-Driven Climate Cycles: A Scientific Framework for Ethiopian Water Security"**

This phase must be treated as a **scientific validation and production-readiness task**, not merely as a coding task.

The objective is to:

1. Test the project's scientific hypotheses using the existing empirical results.
2. Perform a genuinely **blind geographic holdout evaluation** using `eth001.rwl` from Debrebirkan Selassie.
3. Evaluate whether the Random Forest model trained on `eth007` Gondar generalizes to an unseen Ethiopian Highlands site.
4. Preserve strict temporal and geographic separation between training and evaluation data.
5. Produce publication-quality scientific figures and reproducible analysis artifacts.
6. Create a clean prediction service that can be consumed by the existing Node.js/Express application.
7. Execute comprehensive software, data, ML, and scientific QA before declaring the phase complete.

Use the available **Antigravity Awesome Skills** wherever they materially improve repository inspection, Python development, scientific-data processing, notebook execution, testing, debugging, ML validation, code review, or productionization.

---

# 1. FIRST: REPOSITORY AND PROJECT AUDIT

Before modifying anything, inspect the entire relevant repository.

Do not assume filenames, directory structures, variable names, model paths, feature names, or existing implementations.

Locate and inspect:

```text
processed_lagged_data.csv
lag_correlation_results.csv
Model.ipynb
eth007 / Gondar training data
africa/eth001.rwl
SPEIbase NetCDF file
existing detrending implementation
existing 11-year smoothing implementation
Random Forest training implementation
saved Random Forest model
existing tests
Python dependency configuration
Node.js/Express application
```

Also inspect:

```text
pyproject.toml
requirements.txt
environment.yml
package.json
```

or their project equivalents.

Determine:

- Python version
- pandas version
- numpy version
- scipy version
- scikit-learn version
- xarray version
- netCDF4 version
- matplotlib version
- pytest version
- model serialization format
- existing project architecture

Do not duplicate functionality that already exists.

If the project already has reusable detrending, smoothing, feature-engineering, model-loading, or data-validation functions, reuse them.

---

# 2. CRITICAL SCIENTIFIC INTEGRITY RULE

The `eth001` Debrebirkan Selassie dataset is the **geographic holdout dataset**.

It must remain completely unseen during model development.

The holdout dataset must NOT be used for:

- model fitting,
- model refitting,
- hyperparameter tuning,
- feature selection,
- lag selection,
- threshold selection,
- model calibration,
- choosing between competing models,
- modifying the trained model,
- selecting preprocessing parameters,
- determining which features to retain.

The trained Random Forest must be loaded exactly as produced from the `eth007` Gondar training workflow.

The holdout evaluation must be:

```text
LOAD MODEL
    ↓
LOAD HOLDOUT DATA
    ↓
GENERATE REQUIRED FEATURES
    ↓
ALIGN FEATURES
    ↓
PREDICT
    ↓
EVALUATE
```

Never:

```text
LOAD HOLDOUT
    ↓
FIT / TUNE MODEL
```

If the existing pipeline makes this impossible, stop and report the architectural problem rather than silently introducing leakage.

---

# 3. PHASE 4 DELIVERABLES

The final implementation must produce four major deliverables:

### Deliverable A — Scientific Hypothesis Analysis

- hypothesis evaluation,
- publication-quality figures,
- lag interpretation,
- scientific discussion in `Model.ipynb`.

### Deliverable B — Geographic Holdout Validation

- Debrebirkan SPEI extraction,
- `eth001.rwl` feature extraction,
- feature/target alignment,
- blind Random Forest inference,
- classification metrics,
- confusion matrix,
- holdout prediction CSV.

### Deliverable C — Production Prediction Service

A reusable:

```text
predict_service.py
```

that exposes prediction functionality to the existing application.

### Deliverable D — Engineering QA Report

A reproducible report covering:

- data QA,
- leakage QA,
- model QA,
- scientific QA,
- software QA,
- test results,
- holdout performance,
- limitations,
- recommendations.

---

# 4. HYPOTHESIS 1 — SOLAR–DROUGHT CONNECTION

Evaluate:

> **Hypothesis 1:** Periods of high solar activity statistically correspond to below-average tree-ring growth and increased drought incidence.

Use the existing:

```text
processed_lagged_data.csv
lag_correlation_results.csv
```

Do not recompute existing results unless necessary to verify them.

First inspect the actual columns and confirm the meaning of each feature.

Do not assume column names beyond those actually present in the repository.

---

# 5. HYPOTHESIS 1 — REQUIRED ANALYSIS

Generate a publication-quality `matplotlib` time-series visualization covering:

```text
1874–2009
```

The primary comparison should be:

```text
standardized RWI (RWI_z)
vs.
standardized Sunspot activity (SN_z)
```

Use a scientifically appropriate visualization.

If a dual-axis chart is used:

- clearly label both axes,
- clearly identify units,
- use an appropriate legend,
- include the temporal range,
- use readable typography,
- avoid misleading scaling,
- ensure the figure is suitable for publication.

Do not imply causality from visual correlation.

---

# 6. HISTORICAL DROUGHT YEARS

Inspect the project's existing historical drought-year source/documentation.

Do not invent historical drought years.

If the project already contains an authoritative list, use it.

If the repository does not contain a documented drought-year source, report that limitation rather than silently creating one.

When documented severe drought years are available:

- highlight them on the figure,
- clearly distinguish them from ordinary years,
- explain the source in the notebook.

Do not label a year as a severe drought year merely because the SPEI threshold happens to be below `-1.5` unless that is explicitly the project's definition.

---

# 7. HYPOTHESIS 1 — STATISTICAL INTERPRETATION

Use the existing correlation analysis where appropriate.

The analysis must distinguish:

```text
association
```

from:

```text
causation
```

Do not state that solar activity causes drought based solely on correlation.

Discuss:

- direction of association,
- magnitude,
- statistical significance,
- temporal alignment,
- uncertainty,
- sample size,
- possible confounders,
- limitations of observational inference.

If the evidence is weak, state that explicitly.

Do not force the results to support the hypothesis.

---

# 8. HYPOTHESIS 2 — BIOLOGICAL DELAY

The project's empirical lag analysis produced:

```text
optimal lag τ* = 0 years
R = +0.1970
p = 0.0215
```

Before using these values, verify them against:

```text
lag_correlation_results.csv
```

If the actual file contains different values, use the actual calculated values and explain the discrepancy.

Do not hard-code the supplied values if the dataset contradicts them.

---

# 9. HYPOTHESIS 2 — SCIENTIFIC DISCUSSION

Update:

```text
Model.ipynb
```

with an explicit Markdown discussion explaining the implications of an empirical:

```text
τ* = 0
```

rather than the hypothesized:

```text
1–3 year delay
```

Discuss the biological interpretation carefully.

The discussion should address:

- immediate annual radial-growth response,
- climatic forcing and cambial activity,
- physiological response timescales,
- nonstructural carbohydrate storage,
- possible carbohydrate carry-over effects,
- whether storage could blur or mask a delayed relationship,
- why a zero-year statistical lag does not necessarily prove instantaneous biological causation,
- potential effects of annual aggregation,
- limitations of interpreting tree-ring lag correlations biologically.

Do not present speculative physiological mechanisms as established facts.

Clearly distinguish:

```text
empirical result
```

from:

```text
biological interpretation
```

and:

```text
hypothesis for future research
```

---

# 10. NETCDF SPEI HOLDOUT EXTRACTION

Load the regional SPEIbase NetCDF using:

```python
xarray
```

Do not assume its schema.

Inspect:

- dimensions,
- coordinates,
- data variables,
- metadata,
- temporal coverage,
- calendar,
- units,
- missing values.

Verify the intended SPEI-1 variable.

Expected variable:

```text
spei01
```

but verify this before use.

---

# 11. DEBREBIRKAN SELASSIE LOCATION

Holdout site:

```text
eth001.rwl
Debrebirkan Selassie

Latitude ≈ 9.63
Longitude ≈ 39.53
```

Extract the nearest grid cell using the dataset's actual coordinate system.

The intended operation is equivalent to:

```python
.sel(
    lat=9.63,
    lon=39.53,
    method="nearest",
)
```

but first verify coordinate names and longitude convention.

After extraction, record:

```text
requested latitude
requested longitude
selected latitude
selected longitude
latitude offset
longitude offset
```

Do not claim the grid cell is the exact site location.

---

# 12. SPEI ANNUAL AGGREGATION

The source SPEI data are monthly.

Convert to annual SPEI using the project's specified annual aggregation:

```python
.resample(time="YE").mean()
```

or the current equivalent supported by the installed xarray version.

The scientific meaning must remain:

> arithmetic mean of monthly SPEI-1 values within each calendar year.

Before aggregation:

- inspect missing months,
- inspect incomplete years,
- inspect fill values,
- inspect NaN values.

Do not silently treat incomplete years as valid annual observations.

Document the policy used.

---

# 13. DROUGHT CLASSIFICATION

Create:

```text
drought_class
```

using exactly:

```text
Class 0:
SPEI > -1.0

Class 1:
-1.5 < SPEI <= -1.0

Class 2:
SPEI <= -1.5
```

Implement the boundaries explicitly.

Verify the edge cases:

```text
SPEI = -1.0
SPEI = -1.5
```

The expected result is:

```text
-1.0  → Class 1
-1.5  → Class 2
```

Do not leave interval behavior to accidental defaults.

---

# 14. ETH001 TREE-RING PROCESSING

Process:

```text
africa/eth001.rwl
```

using the project's existing scientific pipeline.

Do not create a second, inconsistent detrending implementation.

The processing must reproduce the methodology used for the training site.

The pipeline should include the same relevant steps used for `eth007`, including:

- Tucson `.rwl` parsing,
- raw ring-width extraction,
- biological detrending,
- negative exponential growth curve where that is the established project methodology,
- RWI calculation,
- annual alignment,
- 11-year centered smoothing where required.

The holdout site's feature engineering must be methodologically consistent with the training site's feature engineering.

Do not alter the methodology for the holdout merely to improve performance.

---

# 15. FEATURE CONSISTENCY — CRITICAL

Inspect the exact feature set used when training the saved Random Forest.

The holdout feature matrix must have:

```text
identical feature names
identical feature meaning
identical feature ordering
identical units/scaling
identical lag definitions
```

as the training model expects.

Before prediction, verify:

```python
set(training_features) == set(holdout_features)
```

and also verify ordering.

If there is a mismatch:

**do not guess.**

Stop prediction and report exactly which features are missing or unexpected.

---

# 16. SOLAR FEATURES

Align `eth001` features with the historical sunspot dataset using the same methodology as training.

Use the same:

- annual aggregation,
- smoothing,
- standardization,
- lag definitions,
- temporal alignment.

Do not recompute feature scaling using holdout statistics if the trained model expects standardized features based on training-data statistics.

Do not fit a new scaler on `eth001`.

If the original training pipeline did not persist preprocessing parameters, inspect the architecture and determine how the trained feature representation was generated.

Do not invent a new scaling strategy.

---

# 17. HOLDOUT TEMPORAL ALIGNMENT

Create a final holdout analysis dataset containing, at minimum:

```text
year
model features
SPEI
drought_class
```

Align:

```text
RWI
solar features
SPEI target
```

on `year`.

Validate:

- duplicate years,
- missing years,
- temporal overlap,
- NaN features,
- NaN targets,
- infinite values.

Report:

```text
first holdout year
last holdout year
number of aligned years
number of dropped years
reason for dropped years
```

Do not silently drop observations.

---

# 18. LOAD THE TRAINED RANDOM FOREST

Locate the model artifact produced by the existing `eth007` training pipeline.

Load it without refitting.

The holdout evaluation must use:

```python
model.predict(X_holdout)
```

and:

```python
model.predict_proba(X_holdout)
```

where supported.

Do not call:

```python
model.fit(...)
```

during holdout evaluation.

Do not use `eth001` observations to modify the model.

---

# 19. MODEL PROVENANCE CHECK

Before inference, record:

```text
model path
model serialization format
training dataset/site
model hyperparameters
feature list
training period
model version if available
```

The final report must explicitly state:

> The Random Forest used for geographic holdout evaluation was trained solely on the `eth007` Gondar dataset and was not refit using `eth001` Debrebirkan data.

Only make this statement if verified from the repository.

---

# 20. BLIND HOLDOUT INFERENCE

Run:

```python
predicted_class = model.predict(X_holdout)
probabilities = model.predict_proba(X_holdout)
```

Do not optimize the prediction threshold using the holdout data.

For each year, retain:

```text
year
actual_class
predicted_class
prob_class_0
prob_class_1
prob_class_2
```

Use explicit class ordering.

Verify that probability rows sum approximately to:

```text
1.0
```

within an appropriate floating-point tolerance.

---

# 21. HOLDOUT OUTPUT

Export:

```text
holdout_validation_results.csv
```

with:

```text
year
actual_class
predicted_class
prob_class_0
prob_class_1
prob_class_2
```

If the existing project requires a different confidence-column naming convention, preserve compatibility while documenting the schema.

Requirements:

- chronological order,
- no duplicate years,
- no pandas index,
- numeric probabilities,
- probabilities between 0 and 1,
- probability sum ≈ 1,
- no NaN values.

Read the CSV back from disk and validate it before declaring success.

---

# 22. HOLDOUT CLASSIFICATION REPORT

Generate:

```python
classification_report(
    y_true,
    y_pred,
)
```

Report:

```text
Class 0 precision
Class 0 recall
Class 0 F1
Class 0 support

Class 1 precision
Class 1 recall
Class 1 F1
Class 1 support

Class 2 precision
Class 2 recall
Class 2 F1
Class 2 support

accuracy
macro average
weighted average
```

Also calculate:

```text
balanced accuracy
macro F1
weighted F1
```

Do not report only accuracy.

---

# 23. HOLDOUT CONFUSION MATRIX

Generate the confusion matrix with explicit class ordering:

```text
0 = Normal / Wet
1 = Moderate Drought
2 = Severe Drought
```

Produce:

1. raw count matrix,
2. normalized matrix where appropriate,
3. publication-quality visualization.

Clearly label:

```text
rows = actual
columns = predicted
```

---

# 24. SEVERE DROUGHT PERFORMANCE

Explicitly evaluate:

```text
Class 2 — Severe Drought
```

Report:

```text
precision
recall
F1
support
```

This is especially important for the project's water-security objective.

A model with high overall accuracy but poor severe-drought recall must not be described as operationally successful.

---

# 25. HOLDOUT PERFORMANCE INTERPRETATION

Interpret the geographic holdout as a test of:

> **spatial generalization from Gondar (`eth007`) to Debrebirkan Selassie (`eth001`).**

Do not describe it as ordinary test-set performance.

Discuss:

- whether performance degrades relative to training/cross-validation,
- whether severe drought remains detectable,
- whether errors are concentrated in a class,
- whether temporal or climatic distribution shift may explain degradation.

Do not overstate generalization from a single holdout site.

Explicitly state:

> One geographic holdout site does not establish nationwide Ethiopian generalization.

---

# 26. FEATURE IMPORTANCE

Use the trained model's feature importance only for interpretability.

If the model contains:

```python
feature_importances_
```

extract it.

Ensure feature names are correctly aligned.

Create:

```text
feature
importance
```

and rank descending.

If multiple training models exist from cross-validation, inspect the existing validation architecture and use the most scientifically defensible importance summary.

Do not claim feature importance demonstrates causality.

Do not interpret Random Forest impurity importance as evidence that a feature physically causes drought.

---

# 27. FEATURE IMPORTANCE REPORT

Provide a publication-quality feature-importance visualization.

Clearly distinguish:

```text
predictive importance
```

from:

```text
physical/causal importance
```

Discuss whether solar lag features, RWI features, or other variables dominate prediction.

If the feature importance contradicts the project's scientific hypothesis, report the contradiction rather than hiding it.

---

# 28. PRODUCTION MICROSERVICE

Create:

```text
predict_service.py
```

following existing project architecture.

The service must expose a clean Python API.

Primary function:

```python
predict_drought(
    latitude: float,
    longitude: float,
    year: int,
)
```

Return a JSON-serializable object containing:

```json
{
  "predicted_drought_class": 0,
  "severity_label": "Normal",
  "confidence_probabilities": {
    "class_0": 0.00,
    "class_1": 0.00,
    "class_2": 0.00
  }
}
```

Use the actual model probabilities.

Do not return fabricated confidence values.

---

# 29. SERVICE INPUT VALIDATION

Validate:

```text
latitude
longitude
year
```

Requirements:

```text
latitude ∈ [-90, 90]
longitude ∈ [-180, 180]
year is a valid integer
```

If the project requires 0–360 longitude handling, normalize it consistently with the NetCDF coordinate system.

Reject invalid input with a clear error.

Do not silently convert invalid coordinates into a different location.

---

# 30. SERVICE FEATURE GENERATION

For:

```text
latitude
longitude
year
```

the service must:

1. Load/reuse the local SPEI NetCDF resources appropriately.
2. Select the nearest spatial grid cell.
3. Obtain the required solar/tree-ring feature representation.
4. Construct exactly the feature schema expected by the saved Random Forest.
5. Run inference.
6. Return the prediction.

However, carefully distinguish between:

```text
features required to make a historical prediction
```

and:

```text
features that cannot actually be known for a future year.
```

Do not claim the service provides genuine real-time forecasting if the required RWI feature for the requested year would only become available after tree growth has occurred.

If the current architecture is retrospective rather than predictive, clearly document this limitation.

---

# 31. MODEL LOADING

The service must not retrain the model on every request.

Load the model once where the application architecture permits.

Prefer a reusable service/model singleton or equivalent architecture.

Do not repeatedly read large NetCDF/model files unnecessarily for every request.

Use caching where appropriate without creating stale-data bugs.

---

# 32. SERVICE ERROR HANDLING

Handle:

- missing model artifact,
- missing NetCDF file,
- invalid coordinates,
- year outside available data,
- missing feature data,
- feature-schema mismatch,
- malformed model artifact.

Errors must be:

- explicit,
- actionable,
- loggable,
- safe for the calling application.

Do not expose internal filesystem paths or stack traces to an external API response unless the project explicitly requires development-mode behavior.

---

# 33. OPTIONAL HTTP WRAPPER

If the existing Node.js/Express architecture requires an HTTP boundary, implement a lightweight Python API wrapper using the project's approved dependencies.

Do not introduce a large framework unnecessarily.

The API should provide an endpoint conceptually equivalent to:

```text
POST /predict
```

with:

```json
{
  "latitude": 9.63,
  "longitude": 39.53,
  "year": 2009
}
```

and return the prediction JSON.

Only implement the HTTP layer if it integrates naturally with the existing architecture.

Do not create an unnecessary second API architecture.

---

# 34. API CONTRACT

Document:

### Request

```text
latitude: float
longitude: float
year: int
```

### Response

```text
predicted_drought_class: integer
severity_label: string
confidence_probabilities: object
```

Class mapping:

```text
0 → Normal
1 → Moderate Drought
2 → Severe Drought
```

The response must always contain probabilities for all three classes when the model supports them.

---

# 35. TESTING — MANDATORY

Run the full project test suite:

```bash
pytest
```

Do not stop after one successful script execution.

Add tests for:

### Model loading

Verify the model loads successfully.

### Feature schema

Verify service-generated features match the model's expected schema.

### Prediction

Verify predictions are one of:

```text
0
1
2
```

### Probabilities

Verify:

```text
0 <= probability <= 1
```

and:

```text
sum(probabilities) ≈ 1
```

### Class labels

Verify:

```text
0 → Normal
1 → Moderate Drought
2 → Severe Drought
```

### Input validation

Test invalid:

- latitude,
- longitude,
- year.

### Holdout pipeline

Test the complete:

```text
NetCDF
→ SPEI
→ RWI
→ solar features
→ alignment
→ model
→ prediction
```

pipeline using controlled fixtures where possible.

---

# 36. LEAKAGE RED-TEAM TESTING

Actively attempt to introduce leakage.

Verify that `eth001` does not appear in:

- training data,
- model fitting,
- preprocessing fitting,
- feature selection,
- hyperparameter tuning.

Verify that holdout predictions are generated using:

```text
model.predict()
```

without:

```text
model.fit()
```

during the holdout stage.

Search the implementation for accidental fitting operations.

This is a mandatory QA step.

---

# 37. TEMPORAL LEAKAGE AUDIT

Inspect all lag and smoothing operations.

Verify that:

```text
lag 0
lag 1
...
lag 5
```

have the same definitions used during training.

Pay particular attention to centered 11-year smoothing.

Determine whether centered smoothing introduces future information relative to the prediction year.

This is scientifically critical.

If centered smoothing uses:

```text
t-5 ... t ... t+5
```

then a feature for year `t` contains future observations.

Do not ignore this.

Determine whether the project's model is:

```text
retrospective reconstruction
```

or:

```text
true forward prediction
```

and document the implication.

If the project claims operational forecasting while using future-dependent centered features, flag this as a **critical methodological limitation**.

Do not silently "fix" the scientific methodology without documenting the change.

---

# 38. DATA LEAKAGE AUDIT

Check:

```text
SPEI
drought_class
future RWI
future solar values
future-smoothed values
future-derived lag features
```

Ensure none are incorrectly used as predictors for a genuinely prospective prediction.

Report:

```text
target leakage = PASS/FAIL
temporal leakage = PASS/FAIL
geographic leakage = PASS/FAIL
preprocessing leakage = PASS/FAIL
feature leakage = PASS/FAIL
```

Only mark PASS when actually verified.

---

# 39. REPRODUCIBILITY

The entire Phase 4 analysis must be reproducible.

Record:

```text
dataset paths
dataset versions where available
model artifact
model parameters
feature list
coordinate selection
SPEI aggregation method
drought thresholds
random seed
Python version
package versions
```

Where practical, save a metadata JSON artifact.

Do not modify raw source datasets.

---

# 40. NOTEBOOK EXECUTION

Update:

```text
Model.ipynb
```

with all required Phase 4 analysis.

The notebook must contain sections:

```text
1. Project Objective
2. Dataset Audit
3. Hypothesis 1
4. Solar/RWI Visualization
5. Historical Drought Analysis
6. Hypothesis 1 Interpretation
7. Lag Correlation Results
8. Hypothesis 2 Biological Interpretation
9. Debrebirkan SPEI Extraction
10. eth001 Tree-Ring Processing
11. Holdout Feature Alignment
12. Model Loading
13. Blind Geographic Holdout Prediction
14. Classification Report
15. Confusion Matrix
16. Severe Drought Analysis
17. Feature Importance
18. Holdout Limitations
19. Scientific Conclusions
20. Engineering QA
21. Recommendations
```

Execute the notebook:

```text
from a completely clean kernel
```

Run every cell top-to-bottom.

There must be:

```text
no hidden variables
no execution-order dependencies
no failed cells
no missing files
no fabricated outputs
```

---

# 41. PUBLICATION-QUALITY FIGURES

All final scientific figures must:

- use `matplotlib`,
- have clear axis labels,
- have descriptive titles,
- have legends where necessary,
- use readable fonts,
- use appropriate figure dimensions,
- use tight layout,
- avoid clipped labels,
- include scientifically meaningful annotations,
- be saved at publication-appropriate resolution.

Save figures under the project's existing output convention, or use:

```text
outputs/figures/
```

where no convention exists.

At minimum generate:

```text
solar_rwi_hypothesis.png
holdout_confusion_matrix.png
feature_importance.png
```

Add other figures when scientifically useful.

---

# 42. OUTPUT ARTIFACTS

Generate, as appropriate:

```text
outputs/
├── figures/
│   ├── solar_rwi_hypothesis.png
│   ├── holdout_confusion_matrix.png
│   └── feature_importance.png
│
├── validation/
│   ├── holdout_validation_results.csv
│   ├── holdout_classification_report.json
│   ├── holdout_confusion_matrix.csv
│   └── holdout_metrics.json
│
└── metadata/
    └── phase4_metadata.json
```

Follow existing project conventions if they differ.

Do not overwrite important existing results silently.

---

# 43. DATA QUALITY CHECKS

Before final sign-off, inspect:

```text
row counts
year ranges
duplicate years
missing years
NaN values
infinite values
feature ranges
class distributions
probability ranges
probability sums
```

For holdout data, explicitly report:

```text
Class 0 count
Class 1 count
Class 2 count
```

Acknowledge if the holdout sample is too small to support strong statistical conclusions.

---

# 44. MODEL PERFORMANCE SANITY CHECKS

Investigate:

- suspiciously high performance,
- suspiciously low performance,
- class collapse,
- severe drought underprediction,
- probability calibration concerns,
- geographic distribution shift,
- temporal distribution shift.

Do not celebrate high accuracy without examining class-specific performance.

Do not describe poor performance as evidence that the scientific hypothesis is false.

Model failure and hypothesis failure are not automatically equivalent.

---

# 45. SCIENTIFIC CONCLUSION RULES

The final report must clearly separate:

### What the data demonstrate

Direct empirical findings.

### What the model demonstrates

Predictive/generalization findings.

### What remains uncertain

Scientific limitations and alternative explanations.

### What cannot be concluded

Avoid unsupported causal claims.

Do not convert:

```text
correlation
```

into:

```textcausation
```

Do not convert:

```textsingle-site holdout
```

into:

```textnationwide validation
```

Do not convert:

```textpredictive feature importance
```

into:

```textphysical mechanism
```

---

# 46. MLOPS / PRODUCTION READINESS REVIEW

Review `predict_service.py` as though it were being submitted for production.

Check:

```text
configuration
logging
error handling
input validation
model loading
resource management
feature consistency
testability
determinism
performance
security
```

Do not expose:

- secrets,
- credentials,
- internal filesystem information,
- stack traces,
- unnecessary model internals.

Do not hard-code API keys or credentials.

---

# 47. END-TO-END TEST

Run:

```text
Raw eth001 RWL
        ↓
RWL parsing
        ↓
Detrending
        ↓
RWI
        ↓
11-year processing
        ↓
Solar feature alignment
        ↓
SPEI extraction
        ↓
Annual aggregation
        ↓
Drought classification
        ↓
Feature schema validation
        ↓
Saved Random Forest
        ↓
Blind prediction
        ↓
Probabilities
        ↓
Evaluation
        ↓
CSV output
        ↓
Prediction service
```

Verify every stage.

If a stage fails, fix the underlying issue rather than bypassing it.

---

# 48. FINAL PRODUCTION QA GATE

Do not declare Phase 4 complete until the following checklist has been evaluated:

```text
[ ] Repository audited
[ ] Existing architecture reused
[ ] Dependencies verified
[ ] NetCDF schema verified
[ ] SPEI variable verified
[ ] Coordinate system verified
[ ] Actual holdout grid cell recorded
[ ] Time/calendar verified
[ ] Monthly SPEI validated
[ ] Annual aggregation validated
[ ] Incomplete-year policy enforced
[ ] Drought thresholds verified
[ ] eth001 processed with training-consistent methodology
[ ] Feature schema verified
[ ] No holdout refitting
[ ] No geographic leakage
[ ] No temporal leakage
[ ] No preprocessing leakage
[ ] Model artifact verified
[ ] Blind inference completed
[ ] Probabilities validated
[ ] Holdout CSV generated
[ ] Classification report generated
[ ] Confusion matrix generated
[ ] Balanced accuracy calculated
[ ] Macro F1 calculated
[ ] Severe drought performance evaluated
[ ] Feature importance generated
[ ] Hypothesis 1 evaluated
[ ] Hypothesis 2 evaluated
[ ] Biological interpretation documented
[ ] Notebook executes cleanly
[ ] pytest passes
[ ] Unit tests pass
[ ] Integration tests pass
[ ] Red-team tests pass
[ ] Service input validation tested
[ ] Service prediction tested
[ ] API contract documented
[ ] Output artifacts verified
[ ] Reproducibility metadata recorded
```

If any critical item fails, do not declare the project production-ready.

---

# 49. FINAL ENGINEERING REPORT

At the end of the implementation, provide a concise but rigorous engineering report.

Use exactly this structure:

## 1. Executive Summary

Summarize what was completed.

## 2. Files Created / Modified

List exact file paths and what changed.

## 3. Hypothesis 1 Results

Report:

```text
correlation
p-value
direction
statistical interpretation
```

and explain whether the evidence supports, weakly supports, or does not support the hypothesis.

## 4. Hypothesis 2 Results

Report:

```text
optimal lag
R
p-value
```

and provide the biological interpretation.

## 5. Geographic Holdout

Report:

```text
training site
holdout site
holdout coordinates
selected SPEI grid cell
holdout period
number of observations
```

## 6. Holdout Model Performance

Report:

```text
accuracy
balanced accuracy
macro F1
weighted F1
Class 0 F1
Class 1 F1
Class 2 F1
Class 2 recall
```

## 7. Confusion Matrix

Provide the actual matrix.

## 8. Feature Importance

Provide the ranked feature importance results.

## 9. Leakage Audit

Report:

```text
temporal leakage
geographic leakage
target leakage
preprocessing leakage
future-feature leakage
```

with actual PASS/FAIL results and explanations.

## 10. Software QA

Report actual:

```text
pytest
unit tests
integration tests
red-team tests
linting
formatting
type checking
notebook execution
```

Do not fabricate results.

## 11. Prediction Service

Report:

```text
service location
function/API
input schema
output schema
model-loading strategy
error handling
```

## 12. Limitations

Clearly identify scientific, statistical, data, and engineering limitations.

At minimum consider:

- one-site geographic holdout,
- observational correlation,
- annual aggregation,
- centered smoothing,
- possible future-information leakage for prospective forecasting,
- SPEI spatial resolution,
- sample size,
- class imbalance,
- uncertainty in biological interpretation.

## 13. Recommendations

Separate into:

### Critical Before Operational Deployment

### Important Research Improvements

### Future Enhancements

Recommendations must be based on actual findings.

## 14. Final Status

Use exactly one:

```text
PRODUCTION-READY
```

or:

```text
NOT PRODUCTION-READY
```

Do not use `PRODUCTION-READY` merely because all Python files execute.

A scientific or methodological blocker must result in:

```text
NOT PRODUCTION-READY
```

---

# NON-NEGOTIABLE RULES

1. **Do not fabricate data, metrics, test results, scientific evidence, or file contents.**
2. **Inspect the repository before modifying it.**
3. **Do not assume filenames, columns, NetCDF variables, or coordinate names.**
4. **Do not refit the Random Forest using `eth001`.**
5. **Do not tune the model using holdout performance.**
6. **Do not use holdout results to select features or lags.**
7. **Do not use holdout results to select thresholds.**
8. **The `eth001` site is a blind geographic holdout.**
9. **Use exactly the training feature representation expected by the saved model.**
10. **Do not silently alter the feature-generation methodology between training and holdout.**
11. **Do not fit new preprocessing parameters using holdout data.**
12. **Do not introduce temporal leakage through rolling, centered, lagged, or smoothed features.**
13. **Explicitly investigate whether centered 11-year smoothing contains future information.**
14. **Do not claim retrospective reconstruction is equivalent to prospective forecasting.**
15. **Do not interpret correlation as causation.**
16. **Do not interpret feature importance as physical causality.**
17. **Do not interpret one geographic holdout as nationwide Ethiopian validation.**
18. **Do not hide poor severe-drought recall behind overall accuracy.**
19. **Do not silently discard incomplete or invalid climate observations.**
20. **Do not silently modify raw scientific datasets.**
21. **Do not create duplicate scientific-processing implementations when existing ones can be reused.**
22. **Do not train the model inside the prediction service.**
23. **Do not load large resources unnecessarily for every prediction request.**
24. **Do not expose secrets or internal server information.**
25. **Run the complete test suite.**
26. **Run the notebook from a clean kernel.**
27. **Perform leakage red-team testing.**
28. **Perform data-quality validation.**
29. **Perform end-to-end integration testing.**
30. **Report actual results rather than expected results.**
31. **If a methodological problem is discovered, do not conceal it to achieve a passing status.**
32. **If a critical scientific or engineering issue remains unresolved, mark the project `NOT PRODUCTION-READY`.**
33. **Think like both an ML engineer and a scientific researcher whose results must survive independent scrutiny.**