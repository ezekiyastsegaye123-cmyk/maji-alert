# AGY Production Data Engineering Prompt — SPEIbase NetCDF → Annual Ground-Truth Dataset

Act as a **senior data engineer, scientific Python engineer, and data-quality/ML pipeline engineer**.

I have downloaded a regional **SPEIbase NetCDF (`.nc`) dataset** for use as the historical ground-truth climate signal in our Random Forest drought-classification pipeline.

The goal is to extract the SPEI time series corresponding to the **Debrebirkan Selassie** location, convert the monthly SPEI data to an annual time series, validate the result, and export a clean CSV that can be consumed by the existing ML pipeline.

You are working inside my existing **VS Code project**.

Use the available **Antigravity Awesome Skills** wherever they materially improve repository inspection, Python development, scientific-data processing, NetCDF handling, testing, debugging, notebook execution, and data-quality validation.

Do not simply write a one-off script.

Implement this as a **reproducible, validated, production-quality data-ingestion component** that integrates cleanly with the existing project.

---

# 1. Inspect the Repository First

Before modifying anything:

1. Inspect the repository structure.
2. Identify the existing SPEI/Random Forest pipeline.
3. Identify where input datasets are stored.
4. Identify the project's Python version.
5. Inspect:
   - `pyproject.toml`
   - `requirements.txt`
   - `environment.yml`
   - or the project's equivalent dependency configuration.
6. Inspect existing data-processing modules.
7. Inspect existing tests.
8. Inspect existing Jupyter notebooks.
9. Identify established output-directory and logging conventions.
10. Determine whether `xarray`, `netCDF4`, `numpy`, `pandas`, and related dependencies are already installed/configured.

Do not create duplicate utilities when existing project functionality can be reused.

Do not assume the NetCDF variable names, coordinate names, dimensions, time representation, or calendar until the actual dataset is inspected.

---

# 2. Environment and Dependency Management

The required packages are:

```text id="9cvh0u"
xarray
netCDF4
pandas
numpy
```

First determine whether `xarray` and `netCDF4` are already available in the project's active virtual environment.

Use the project's existing dependency-management mechanism.

If missing:

- Install them into the active project virtual environment.
- Update the project's dependency configuration where appropriate.
- Do not install packages globally.
- Do not create an unnecessary second virtual environment.
- Do not silently modify system Python.

Verify the final environment by successfully importing:

```python
import xarray as xr
import netCDF4
import pandas as pd
import numpy as np
```

Record the versions used where appropriate for reproducibility.

---

# 3. Locate and Inspect the NetCDF File

Locate the downloaded `.nc` file in the workspace.

Do not assume its filename.

If the file cannot be found, report the exact issue rather than inventing a path.

Open the dataset using xarray.

For example:

```python
import xarray as xr

ds = xr.open_dataset(input_path)
```

Before extracting any data, inspect and report:

- dataset dimensions,
- coordinates,
- data variables,
- variable attributes,
- units,
- calendar/time metadata,
- latitude orientation,
- longitude convention,
- dataset temporal range,
- missing-value representation.

Use diagnostics equivalent to:

```python
ds.dims
ds.coords
ds.data_vars
ds.attrs
```

and inspect the relevant variable metadata.

Do not assume `spei01` exists without verifying it.

---

# 4. Identify the Correct SPEI Variable

The expected monthly variable is:

```text id="edpp21"
spei01
```

Verify that it actually exists.

If the dataset uses a different variable name:

1. Inspect the available variables and attributes.
2. Determine which variable represents the intended 1-month SPEI product.
3. Use the verified variable.
4. Document the mapping.

Do not blindly select the first numeric variable.

Confirm the SPEI variable's:

- units,
- dimensions,
- temporal frequency,
- spatial dimensions,
- missing-value conventions.

The final extracted variable should represent **monthly SPEI-1** values.

---

# 5. Coordinate Verification

The requested extraction location is:

```text
Latitude:  9.63
Longitude: 39.53
Location:  Debrebirkan Selassie
```

The intended extraction logic is:

```python
.sel(
    lat=9.63,
    lon=39.53,
    method="nearest",
)
```

However, before executing this blindly, inspect the actual coordinate system.

Verify:

- latitude coordinate name,
- longitude coordinate name,
- latitude units,
- longitude units,
- longitude convention,
- coordinate ranges,
- whether coordinates are ascending or descending.

For example, determine whether longitude is represented as:

```text
-180 ... +180
```

or:

```text
0 ... 360
```

If the dataset uses `0–360` longitude and `39.53` is already valid, retain it.

If coordinate normalization is necessary, perform it explicitly and document the transformation.

Do not silently select a physically incorrect location because of longitude-convention differences.

---

# 6. Nearest-Grid-Cell Validation

Use nearest-neighbor extraction for the requested location:

```python
spei_point = ds["spei01"].sel(
    lat=9.63,
    lon=39.53,
    method="nearest",
)
```

After extraction, determine the **actual grid-cell coordinates selected**.

Report:

```text id="p6z4yd"
requested_lat
requested_lon
selected_lat
selected_lon
latitude_difference
longitude_difference
```

This is mandatory.

Calculate the spatial distance or at minimum coordinate offsets between the requested point and the selected grid cell.

Do not claim that the dataset represents the exact Debrebirkan Selassie location if the grid cell is only approximately nearest.

The final report should explicitly state the selected grid cell.

---

# 7. Time Coordinate Validation

Inspect the NetCDF time coordinate carefully.

Do not assume it is a normal pandas `datetime64` index.

SPEIbase may use standard or non-standard CF calendars, so determine:

- time dtype,
- calendar,
- frequency,
- start date,
- end date,
- missing timestamps,
- duplicate timestamps.

Confirm that the source data are monthly.

The implementation must correctly handle CF-compliant time representations.

Do not convert dates using fragile string slicing if xarray's decoded time representation is available.

---

# 8. Monthly-to-Annual Aggregation

The raw `spei01` data are monthly.

Convert them to an annual time series.

The intended aggregation is the **arithmetic mean of the monthly SPEI values within each calendar year**.

Use an xarray-compatible annual resampling operation equivalent to:

```python
.resample(time="YE").mean()
```

However, verify the behavior of the installed xarray version.

Do not assume deprecated resampling aliases.

If the project's xarray version uses a different accepted year-end frequency alias, use the current supported equivalent while preserving the intended meaning:

> calendar-year average ending in December.

Document the exact resampling rule.

---

# 9. Incomplete Years

This is critical.

Before calculating annual means, inspect whether the dataset contains:

- complete years,
- partially missing years,
- months with missing SPEI values.

Do not blindly calculate an annual average from one or two available months and treat it as a complete annual climate value.

Determine an explicit policy.

Preferred validation policy:

- Require a complete 12-month calendar year for the ground-truth annual label.
- Mark incomplete years as invalid for the final ground-truth CSV.
- Report which years were incomplete and removed.

If the project has an established scientific convention for handling partial years, follow it instead and document the rule.

Do not silently change the data.

---

# 10. Missing Values

Inspect the raw monthly SPEI values for:

- NaN,
- inf,
- fill values,
- masked values,
- dataset-specific missing-value encodings.

Use xarray's decoded/metadata-aware behavior appropriately.

After annual aggregation:

- Remove invalid annual observations from the final exported dataset.
- Report how many observations were removed.
- Report which years were affected.

Do not convert missing values to zero.

Do not interpolate missing SPEI unless explicitly authorized.

Do not fabricate values.

---

# 11. Annual DataFrame Construction

Convert the validated annual series to a pandas DataFrame.

The final DataFrame must contain exactly:

```text id="gldxgu"
year
spei
```

Requirements:

- `year` must be an integer year.
- `spei` must be numeric.
- Sort chronologically.
- One row per calendar year.
- No duplicate years.
- No pandas index column in the CSV.
- No unnamed columns.

Example:

```text id="qji5bi"
   year      spei
0  1874   ...
1  1875   ...
2  1876   ...
...
```

The exact values must come from the actual NetCDF dataset.

---

# 12. Year-Level Integrity Checks

Before writing the CSV, verify:

```text id="rzzf2z"
✓ year is numeric/integer
✓ year is strictly increasing
✓ no duplicate years
✓ no missing SPEI values
✓ SPEI values are finite
✓ expected annual frequency
✓ output contains exactly two columns
```

Also inspect year gaps.

If the final output contains:

```text
1900
1901
1903
```

do not silently create `1902`.

Report the gap.

---

# 13. Temporal Range Validation

Print and record:

```text id="p5tcvj"
first year
last year
number of annual observations
number of missing calendar years
```

Compare the available range with the period required by the downstream tree-ring/RWI analysis.

The downstream model currently references:

```text id="b5bgya"
1874–2009
```

Do not assume the SPEI dataset covers this entire period.

Explicitly determine:

```text overlap with 1874–2009
years missing from overlap
```

This is important because the Random Forest pipeline must not assume that all years have ground-truth labels.

---

# 14. Consistency With Downstream ML Pipeline

Inspect the existing downstream Random Forest pipeline.

Verify that the exported CSV:

```text id="3kgn1r"
spei_debrebirkan.csv
```

can be merged with:

```text id="a85v1q"
processed_lagged_data.csv
```

using:

```text id="0zq7kw"
year
```

The output must therefore use exactly:

```text id="2qf8cb"
year
spei
```

with no hidden index.

Where practical, run a compatibility check:

```python
rwi_df.merge(
    spei_df,
    on="year",
    how="inner",
)
```

and report the number of overlapping years.

Do not modify the RWI/solar dataset merely to make the merge work.

---

# 15. Script Architecture

Do not put the entire workflow in one function.

Use reusable components such as:

```python
load_netcdf()
inspect_dataset()
resolve_spei_variable()
validate_coordinates()
extract_point_series()
validate_time_axis()
aggregate_annual_spei()
validate_annual_data()
to_dataframe()
export_csv()
```

Adapt names to the existing project architecture.

Separate:

```text
data loading
scientific extraction
validation
transformation
export
```

Use type hints and clear docstrings.

Use `pathlib.Path` for filesystem operations.

Do not hard-code machine-specific absolute paths.

---

# 16. CLI

If the project does not already have a suitable command-line interface, provide one.

Example:

```bash
python -m <module> \
    --input path/to/dataset.nc \
    --output path/to/spei_debrebirkan.csv \
    --lat 9.63 \
    --lon 39.53
```

Provide:

```text id="g8e4kn"
--input
--output
--lat
--lon
```

The location should be configurable rather than hard-coded inside the processing function.

The default values may be:

```text
lat = 9.63
lon = 39.53
```

but they should remain overridable.

Provide useful `--help`.

---

# 17. Logging

Use the project's logging conventions.

At appropriate levels report:

```text id="y8t7ih"
Input NetCDF file
Dataset dimensions
SPEI variable selected
Requested coordinates
Selected grid-cell coordinates
Temporal range
Monthly observation count
Annual observation count
Incomplete years
Missing years
Output path
```

Do not print thousands of data values.

---

# 18. Output

Write:

```text id="z2gqvn"
spei_debrebirkan.csv
```

with exactly:

```csv
year,spei
```

Requirements:

- UTF-8.
- No index.
- Stable column order.
- Stable chronological ordering.
- Numeric SPEI values.
- No NaN.
- No infinity.
- No extra columns.

Create the destination directory if necessary, following project conventions.

Do not overwrite existing output silently.

Use an explicit overwrite policy if needed.

---

# 19. Console Verification

After writing the CSV, read the generated CSV back from disk.

Do not simply print the DataFrame before saving and assume the file is correct.

Print:

```text
First 5 rows:
```

followed by the first five rows from the **actual generated CSV**.

Also print a concise summary such as:

```text
Output file: ...
Rows: ...
Year range: ...–...
Selected grid cell: (..., ...)
```

This confirms serialization and deserialization worked correctly.

---

# 20. Jupyter Notebook

Create or update an appropriate notebook in the project, for example:

```text id="rdz1wu"
notebooks/spei_debrebirkan_extraction.ipynb
```

The notebook must use the reusable Python functions instead of duplicating the complete implementation.

Organize it approximately as:

```text
1. Objective
2. Environment
3. Load NetCDF
4. Inspect Dataset
5. Identify SPEI Variable
6. Inspect Coordinates
7. Extract Debrebirkan Selassie Grid Cell
8. Validate Selected Grid Cell
9. Inspect Time Coordinate
10. Validate Monthly Coverage
11. Aggregate to Annual SPEI
12. Validate Complete Years
13. Convert to DataFrame
14. Inspect First 5 Rows
15. Export CSV
16. Re-read CSV
17. Validate Output
18. Check Compatibility With RWI Dataset
19. Summary
```

Execute the notebook from a clean kernel.

It must run top-to-bottom without hidden state.

---

# 21. Optional Scientific Visualization

Where useful, create a simple time-series plot of the final annual SPEI:

```text
year vs SPEI
```

Clearly label:

```text
Year
Annual mean SPEI
```

Add drought reference thresholds if consistent with the downstream classification scheme:

```text
SPEI = -1.0
SPEI = -1.5
```

Do not imply that the plot itself represents drought-class predictions.

Keep visualization separate from the production extraction logic.

---

# 22. Automated Tests

Create tests using the project's existing testing framework.

At minimum test:

### File validation

- missing NetCDF file,
- unreadable file,
- empty/invalid dataset.

### Variable validation

- expected SPEI variable exists,
- alternate variable naming is handled only when verified,
- unrelated variables are not accidentally selected.

### Coordinate extraction

Use a synthetic xarray dataset and verify:

```python
.sel(lat=9.63, lon=39.53, method="nearest")
```

selects the intended grid cell.

Test coordinate order.

Test longitude handling where appropriate.

### Time handling

Test:

- standard monthly dates,
- year boundaries,
- missing months,
- incomplete years,
- duplicate timestamps where applicable.

### Annual aggregation

Verify a synthetic 12-month dataset produces the mathematically expected annual mean.

Example:

```text id="uvd8fj"
12 monthly values → known annual average
```

### Missing values

Verify incomplete/invalid observations are handled according to the documented policy.

### DataFrame schema

Verify:

```text id="a0tvkt"
columns == ["year", "spei"]
```

and:

- no duplicate years,
- no NaN,
- no infinity.

### CSV export

Write to a temporary directory and verify the generated file can be read back correctly.

---

# 23. End-to-End Integration Test

Create an end-to-end synthetic NetCDF fixture.

The test must execute:

```text id="wmb4l5"
synthetic NetCDF
↓
xarray load
↓
variable validation
↓
coordinate extraction
↓
monthly validation
↓
annual aggregation
↓
DataFrame
↓
CSV export
↓
CSV reload
```

Verify the final values against known expected results.

Do not make the real historical SPEIbase file the only test.

---

# 24. Data Engineering Red-Team Testing

Actively attempt to break the extractor.

Test:

- latitude descending instead of ascending,
- longitude 0–360,
- longitude -180–180,
- missing months,
- missing years,
- leap years,
- non-standard CF calendar,
- NaN values,
- fill values,
- unexpected variable names,
- duplicate timestamps,
- empty spatial selection,
- target coordinate outside dataset bounds,
- multiple spatial dimensions,
- malformed output paths.

The system must either handle these cases correctly or fail with a clear actionable error.

Never silently produce plausible-looking but incorrect climate data.

---

# 25. Scientific QA

After extraction, inspect:

```text id="nc0g5j"
selected grid-cell location
SPEI units
annual aggregation rule
number of months per year
number of complete years
missing years
year range
SPEI range
mean
standard deviation
```

Check for suspicious values.

Do not impose arbitrary SPEI validity limits without evidence from the dataset metadata or project requirements.

Do not silently clip SPEI values.

---

# 26. Production Data Provenance

Create or record metadata sufficient to reproduce the extraction:

```text id="kn7n8x"
source NetCDF filename
source variable
requested latitude
requested longitude
selected latitude
selected longitude
selection method
time aggregation method
incomplete-year policy
missing-value policy
output filename
Python version
xarray version
netCDF4 version
pandas version
numpy version
```

Where possible, also retain the relevant source dataset metadata/attributes.

Do not alter the original NetCDF.

---

# 27. Downstream Ground-Truth Validation

Because this dataset will become the target source for the Random Forest model, perform a downstream compatibility check.

Load:

```text id="2bqet7"
processed_lagged_data.csv
```

and the generated:

```text id="69adku"
spei_debrebirkan.csv
```

Merge on `year` using an analysis-only DataFrame.

Report:

```text id="xpssmn"
RWI/solar observations
SPEI observations
overlapping observations
first overlapping year
last overlapping year
years missing from either dataset
```

Do not modify either source dataset.

This check must confirm that the SPEI file is actually usable as the ground-truth source for the Day 3 model.

---

# 28. ML Leakage Consideration

The generated SPEI dataset is intended to define the target.

Therefore:

> SPEI must be treated as the source of the target label and must not be accidentally included as a model predictor.

Inspect the downstream Random Forest feature-construction code and verify that:

```text id="j2y0sa"
spei
```

is not included in `X`.

Also verify that:

```text id="8q3t9v"
drought_class
```

is derived from SPEI only after the datasets are merged.

Do not modify model behavior unless necessary, but flag any leakage discovered.

---

# 29. Quality Gates

Before declaring completion, verify:

```text id="spvh6a"
[ ] NetCDF file located
[ ] Dataset opens successfully
[ ] SPEI variable verified
[ ] Variable frequency verified as monthly
[ ] Coordinates verified
[ ] Requested location validated
[ ] Actual selected grid cell recorded
[ ] Time/calendar validated
[ ] Monthly missing values inspected
[ ] Complete-year policy enforced
[ ] Annual aggregation verified
[ ] Year column validated
[ ] No duplicate years
[ ] No NaN SPEI values in final CSV
[ ] No infinite SPEI values
[ ] Output schema exactly year,spei
[ ] CSV successfully reloaded
[ ] First 5 rows verified from actual CSV
[ ] Downstream merge compatibility checked
[ ] Unit tests pass
[ ] Integration test passes
[ ] Red-team tests completed
[ ] Notebook executes from clean state
[ ] Provenance recorded
```

Do not mark a gate as passed unless it was actually checked.

---

# 30. Final ML/Data Engineering QA

Treat this implementation as though you are reviewing it for inclusion in a production scientific ML pipeline.

Do not stop at "the CSV was generated."

Perform a complete QA review covering:

### Data QA

Check the actual extracted dataset.

### Scientific QA

Confirm the spatial extraction and annual aggregation are correct.

### Software QA

Run the test suite, linting, formatting, and type checking where configured.

### Integration QA

Verify compatibility with the existing RWI/solar dataset.

### Reproducibility QA

Run the extraction twice with the same inputs/configuration and verify deterministic output.

### Notebook QA

Run the notebook from a clean kernel.

### Artifact QA

Read the generated CSV back from disk and inspect it.

### Failure QA

Verify malformed or incomplete inputs fail safely.

---

# 31. Final Recommendations

After testing, provide evidence-based recommendations in three categories.

## Critical Before ML Training

Issues that could invalidate the SPEI ground-truth labels or cause leakage.

## Important

Changes that would improve reliability, maintainability, or scientific reproducibility.

## Future

Optional improvements such as:

- retaining extraction metadata in JSON,
- adding checksums for source datasets,
- adding dataset version/provenance tracking,
- automated schema validation,
- automated data-quality reports,
- pipeline orchestration,
- CI tests,
- reproducible environment locking.

Do not recommend additional complexity unless there is a demonstrated benefit.

---

# 32. Final Report

At completion, provide:

## Implementation Summary

What was implemented.

## Files Created/Modified

Exact paths and responsibilities.

## Dataset Inspection

Report:

```text
NetCDF dimensions:
SPEI variable:
Monthly date range:
Annual date range:
```

## Spatial Extraction

Report:

```text
Requested location:
Selected grid cell:
Latitude offset:
Longitude offset:
Selection method:
```

## Aggregation

Report:

```text
Monthly observations:
Complete annual observations:
Incomplete years:
Aggregation rule:
```

## Output

Report:

```text
Output path:
Rows:
Columns:
Year range:
```

## Verification

Include the actual first five rows read back from the generated CSV.

## Downstream Compatibility

Report:

```text
Overlapping RWI/SPEI years:
First overlap:
Last overlap:
```

## Tests

Report actual:

```text
Unit tests:
Integration tests:
Red-team tests:
Lint:
Formatting:
Type checking:
Notebook execution:
```

Do not fabricate results.

## QA Findings

Identify issues discovered and fixes applied.

## Recommendations

Provide prioritized next steps.

---

# NON-NEGOTIABLE RULES

1. **Do not assume the NetCDF schema. Inspect it first.**
2. **Do not assume `spei01` exists without verifying it.**
3. **Do not assume the coordinate names are `lat` and `lon` without inspection.**
4. **Do not silently mishandle 0–360 versus -180–180 longitude conventions.**
5. **Always report the actual grid cell selected by nearest-neighbor extraction.**
6. **Do not claim the selected cell is the exact physical location unless it is.**
7. **Do not blindly treat incomplete years as valid annual climate observations.**
8. **Do not interpolate missing SPEI values unless explicitly authorized.**
9. **Do not silently convert missing values to zero.**
10. **Do not fabricate missing calendar years.**
11. **Do not modify the source NetCDF.**
12. **Do not modify the raw RWI/solar dataset.**
13. **Do not include a pandas index in the CSV.**
14. **The final CSV must contain exactly `year,spei`.**
15. **Read the generated CSV back from disk before declaring success.**
16. **Use the generated SPEI values as the target source, not as model features.**
17. **Do not introduce temporal or target leakage into the downstream ML pipeline.**
18. **Use the project's dependency-management system.**
19. **Do not install packages globally.**
20. **Create reusable Python code rather than putting everything in a notebook.**
21. **The Jupyter Notebook must use the reusable implementation.**
22. **Run the complete tests.**
23. **Run the notebook from a clean kernel.**
24. **Perform red-team tests against malformed scientific data.**
25. **Do not fabricate test results, dataset values, or QA results.**
26. **Do not call the pipeline production-ready merely because the CSV was created.**
27. **Report scientific and engineering limitations honestly.**
28. **Think like a data engineer responsible for supplying ground-truth labels to a model that other researchers will rely on.**