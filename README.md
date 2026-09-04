# Tree-Ring Detrending Pipeline

A production-ready Python pipeline for reading raw tree-ring width measurements
from standard Tucson-format `.rwl` files, performing biological detrending using
a negative exponential growth model, and calculating standardized Ring Width
Index (RWI).

## Installation

Requires Python ≥ 3.10.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy pandas scipy

# For development/testing
pip install pytest
```

## Usage

### Command Line

```bash
python -m treering <input.rwl> <output.csv> [options]
```

**Arguments:**

| Argument     | Description                          |
| ------------ | ------------------------------------ |
| `input_rwl`  | Path to input `.rwl` file (Tucson format) |
| `output_csv` | Path for output CSV file             |

**Options:**

| Option          | Description                                          |
| --------------- | ---------------------------------------------------- |
| `--overwrite`   | Overwrite existing output CSV file                   |
| `--skip-failed` | Skip series that fail curve fitting (log a warning)  |
| `-v, --verbose` | Enable debug-level logging                           |
| `-h, --help`    | Show help message                                    |

**Examples:**

```bash
# Basic usage
python -m treering africa/eth007.rwl results.csv

# Skip series with too few observations for fitting
python -m treering africa/eth001.rwl results.csv --skip-failed

# Overwrite existing output, verbose logging
python -m treering africa/eth007.rwl results.csv --overwrite --verbose
```

### Python API

```python
from treering import process_rwl, export_csv

# Run the full pipeline
df = process_rwl("africa/eth007.rwl", skip_failed_series=True)
print(df.head())

# Export to CSV
export_csv(df, "output.csv", overwrite=True)
```

## Input Format — Tucson `.rwl`

The pipeline reads standard **Tucson Decadal Format** `.rwl` files as used by
the International Tree-Ring Data Bank (ITRDB).

**Structure:**

- Optional 3-line header (site name, location/species, investigators).
- Data lines: `<series_id>  <decade_start_year>  <value_1> ... <value_N>`
- Series ID: up to 8 characters.
- Values: integer ring widths (units depend on dataset).
- `999` = end-of-series stop marker.
- Multiple series stored sequentially.

**Example:**

```
TST01   1900   500   480   460   440   420   400   380   360   340   320
TST01   1910   300   280   260   250   240   230   220   215   210   205
TST01   1920   200   198   196   999
TST02   1920   400   380   360   340   320   300   285   270   260   250
TST02   1930   240   235   230   999
```

## Processing Pipeline

```
.rwl file
    │
    ▼
┌──────────────┐
│  parse_rwl() │  → Parse Tucson format, extract (series_id, year, ring_width)
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│  For each series:  │
│                    │
│  fit_growth_curve()│  → Fit G(t) = a·exp(-b·t) + c via scipy.optimize.curve_fit
│                    │
│  calculate_rwi()   │  → RWI_t = raw_t / G(t)
└────────┬───────────┘
         │
         ▼
┌──────────────┐
│ export_csv() │  → Write series_id, year, raw_ring_width, fitted_growth, rwi
└──────────────┘
```

### Negative Exponential Growth Model

```
G(t) = a · exp(-b · t) + c
```

| Parameter | Meaning                        | Bounds      |
| --------- | ------------------------------ | ----------- |
| `a`       | Amplitude of decaying component | ≥ 0         |
| `b`       | Decay rate                     | ≥ 0         |
| `c`       | Asymptotic minimum growth      | ≥ 0         |
| `t`       | Age index (year − min(year))   | ≥ 0         |

Fitting uses `scipy.optimize.curve_fit` with bounded parameters.

### Ring Width Index (RWI)

```
RWI_t = RawRingWidth_t / G(t)
```

A well-detrended series has a mean RWI near 1.0. Values > 1.0 indicate
above-average growth; values < 1.0 indicate below-average growth.

## Output Schema

The CSV output contains these columns in order:

| Column           | Type    | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `series_id`      | string  | Tree/core series identifier           |
| `year`           | integer | Calendar year                         |
| `raw_ring_width` | integer | Original measurement from `.rwl` file |
| `fitted_growth`  | float   | Fitted G(t) value                     |
| `rwi`            | float   | Ring Width Index (raw / fitted)       |

- UTF-8 encoded.
- No pandas index column.
- Sorted by `(series_id, year)`.

## Error Handling

The pipeline validates at every stage and provides specific error messages:

| Error                       | Behavior                                            |
| --------------------------- | --------------------------------------------------- |
| Missing input file          | `FileNotFoundError` with path                       |
| Empty / no-data file        | `RWLParseError` with context                        |
| Non-numeric measurements    | `RWLParseError` identifying line and position       |
| Too few observations (<10)  | `FittingError` identifying the series               |
| Non-finite input values     | `FittingError` identifying the series               |
| Curve fit failure           | `FittingError` with scipy error details             |
| Near-zero fitted growth     | `FittingError` or `RWIError` with tolerance info    |
| Non-finite RWI              | `RWIError` identifying the series                   |
| Output file exists          | `ExportError` (use `--overwrite` to replace)        |

With `--skip-failed`, series that fail fitting are logged and omitted rather
than aborting the entire pipeline.

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- Parser: valid files, multi-series, headers, stop markers, edge cases, errors.
- Model: mathematical correctness, parameter recovery from synthetic data.
- RWI: arithmetic correctness, division safety.
- Pipeline: schema validation, independent fitting, real data, determinism.
- Export: column order, index exclusion, overwrite protection.
- End-to-end: `.rwl` → CSV round-trip on both fixtures and real data.

## Assumptions and Limitations

1. **Tucson format variant**: Supports the standard decadal format with
   optional 3-line headers.  Non-standard header formats or non-Tucson
   `.rwl` variants may not parse correctly.

2. **Stop marker**: Only `999` is recognized as an end-of-series marker.
   The missing-value marker `-9999` is not specially handled (treated as a
   regular measurement).

3. **Minimum observations**: At least 10 measurements per series are required
   for curve fitting (3 parameters + margin).

4. **Growth model**: Only the negative exponential model
   `G(t) = a·exp(-b·t) + c` is supported.  Alternative detrending methods
   (spline, linear regression) are not implemented.

5. **Measurement units**: The pipeline does not convert or validate units.
   Ring widths are processed as-is from the `.rwl` file.

## Solar-Cycle Lag Analysis (RWI vs. Sunspot Number)

The package includes a solar-cycle lag analysis module (`treering.solar_lag`) to investigate decadal statistical associations between solar activity (historical Sunspot Number, $SN$) and regional tree growth ($RWI$).

### Analysis Workflow
1. **Merge on Calendar Year**: Exact inner join on integer calendar years $[1700, \dots]$.
2. **11-Year Centered Moving Average**: Isolates the ~11-year Schwabe solar cycle:
   $$x_{\text{smoothed}}(t) = \frac{1}{11} \sum_{k=-5}^{5} x(t+k)$$
   *First 5 and last 5 observations strictly receive `NaN`.*
3. **Standardization**: z-score anomalies with sample standard deviation ($\text{ddof}=1$):
   $$z_t = \frac{x_t - \mu}{\sigma}$$
4. **Lag Correlation**: Evaluates Pearson correlation $R(\tau) = \text{corr}(RWI(t), SN(t-\tau))$ for $\tau \in \{0, 1, 2, 3, 4, 5\}$ years.
5. **Optimal Lag Selection**: $\tau^* = \arg\max_{\tau} |R(\tau)|$ (evaluating strongest positive or negative correlation).

### Python API Example

```python
from treering import run_solar_lag_analysis

# Run end-to-end solar lag analysis
result = run_solar_lag_analysis(
    rwi_input="results/rwi_eth007.csv",
    sunspot_input="SN_y_tot_V2.0.csv",
    max_lag=5,
    output_dir="results",
    overwrite=True,
)

print(f"Optimal lag: tau = {result.optimal_lag.optimal_lag} years")
print(f"Pearson R: {result.optimal_lag.optimal_correlation:.4f}")
print(result.lag_correlations)
```

### Interactive Jupyter Notebook

An interactive notebook with visualizations and step-by-step auditability is available at:
- `notebooks/rwi_sunspot_lag_analysis.ipynb`
- `Model.ipynb`

## Project Structure

```
treering/
├── __init__.py      # Public API exports
├── __main__.py      # python -m treering entry point
├── cli.py           # Argument parsing and CLI logic
├── export.py        # CSV export with validation
├── model.py         # Negative exponential model and curve fitting
├── parser.py        # Tucson .rwl file parser
├── pipeline.py      # End-to-end detrending orchestration
├── rwi.py           # Ring Width Index calculation
└── solar_lag.py     # RWI / Sunspot 11-yr Schwabe lag analysis

notebooks/
└── rwi_sunspot_lag_analysis.ipynb # Interactive analysis & visualization

tests/
├── fixtures/
├── test_model.py
├── test_parser.py
├── test_pipeline.py
├── test_rwi.py
└── test_solar_lag.py

## FRADSCR — Solar Groundwater Pump Drought Warning System

**FRADSCR** is a production web interface and real-time backend API built for low-bandwidth cellular networks in the **Borana Zone, Oromia, Ethiopia**. It links regional tree-ring climate memory (*Juniperus procera*) and solar cycle teleconnections directly to community water point operators and solar-powered borehole pumps.

### Target Architecture

```text
Browser (Mobile / Desktop)
   │
   │ HTTPS / WebSocket (Socket.io)
   ▼
Node.js (Express 5 + Socket.io 4)
   │
   ├── Security Headers (Helmet) & Explicit CORS
   ├── Zod Schema Input Validation (Bounds, types, no injection)
   ├── Rate Limiting & Concurrency Control (Max 3 concurrent ML jobs)
   ├── MongoDB / Mongoose Audit Logging (QueryLog model)
   │
   └── Python ML Engine Integration (predict_service.py)
          │
          │ Local child_process.spawn (shell: false, argv array)
          ▼
     Random Forest Ensemble (Gondar eth007 + SILSO + NetCDF SPEI)
          │
          ▼
     Calibrated 3-Class Drought Output & Probabilities
```

### Key Capabilities
- **Multi-Language Support**: Complete vanilla JavaScript localization in **English**, **Afaan Oromoo** (primary local language of Borana), and **Amharic** (አማርኛ).
- **Zero Framework CDN Overhead**: Pure semantic HTML5, responsive CSS (<12 KB), and vanilla JS. Socket.io is served directly from the backend with zero external CDN dependencies.
- **Borana Zone Presets & Geolocation**: Instant one-tap selection of major pastoral borehole clusters (Yabelo, Dubuluk, Mega, Moyale) or native HTML5 GPS location.
- **Accessible Drought Alert Gauge**: Displays color + icon + explicit bold text (Normal, Moderate Drought, Severe Drought) with pump operation advisories for pastoral herds.
- **Security & Integrity**: Non-shell process invocation (`shell: false`), strict Zod validation at both input and output boundaries, per-socket throttling, and graceful DB failure handling.

### Installation & Quickstart

#### 1. Backend Setup
```bash
# Install Node.js dependencies
npm install

# Copy environment template
cp .env.example .env
```

#### 2. Configure Environment (`.env`)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `3000` | Port for Express & Socket.io server |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017/fradscr` | MongoDB connection string for QueryLog |
| `PYTHON_EXECUTABLE` | `./venv/bin/python` | Path to Python 3.10+ venv binary |
| `ML_SERVICE_PATH` | `./predict_service.py` | Path to prediction service script |
| `ML_TIMEOUT_MS` | `45000` | Execution timeout for Python subprocess |
| `MAX_CONCURRENT_ML_JOBS`| `3` | Maximum concurrent Python ML processes |
| `DEFAULT_YEAR` | `2026` | Default operational projection year |
| `CORS_ORIGIN` | `*` | Allowed CORS origins |

#### 3. Run Server
```bash
# Start production server
npm start

# Or with live logs in development
npm run dev
```

Visit `http://localhost:3000` in any browser or mobile device.

### API Endpoints

#### Health Check
```bash
curl http://localhost:3000/health
```
Returns system status, uptime, database connection state, and ML engine availability without leaking credentials.

#### REST Drought Prediction
```bash
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 4.88, "longitude": 38.08, "year": 2026}'
```

#### Socket.io Real-Time Protocol
- **Request event**: `drought:predict` with `{ latitude: 4.88, longitude: 38.08, year: 2026 }`
- **Status event**: `drought:status`
- **Result event**: `drought:prediction_result` (emitted strictly to requesting socket)
- **Error event**: `drought:prediction_error`

### Automated Testing

```bash
# Run complete Jest test suite (81 tests across unit, integration, and security)
npm test

# Run Python ML test suite (151 tests)
./venv/bin/pytest
```

---

## Production Docker & Reverse Proxy Deployment

FRADSCR is fully containerized and reverse-proxied for high-availability production environments:

### 1. Architectural Topology

```text
Public Internet (HTTPS :443 / HTTP :80)
        │
        ▼
   Nginx Reverse Proxy (DDoS Rate Limiter, SSL/TLS, Gzip Compression)
        │
        ├── /.well-known/acme-challenge/ ──> Certbot (Let's Encrypt Auto-Renewal)
        │
        └── http://web-app:3000 (Internal Docker Bridge Network)
                 │
                 ├── WebSocket / REST API Gateway
                 │
                 ├── http://ml-service:8000 (FastAPI In-Memory ML Engine)
                 └── mongodb:27017 (Audit Query Log)
```

### 2. Start Full Production Stack

```bash
# Build images and launch background services with reverse proxy
docker compose up -d --build
```

This launches 5 coordinated services:
- **`reverse-proxy`**: Nginx 1.25 edge server enforcing rate limiting (`10r/s`), Gzip compression for 2G/3G networks, and secure HTTP/WebSocket proxying (ports `80`, `443`).
- **`web-app`**: Node.js Express 5 + Socket.io application gateway.
- **`ml-service`**: Persistent in-memory FastAPI ML prediction service (isolated on `fradscr_internal` network).
- **`mongodb`**: Audit & query logging store with volume persistence (`mongo_data`).
- **`certbot`**: Automated Let's Encrypt SSL/TLS certificate renewal daemon.

### 3. Verify System Health

```bash
# Check container status
docker compose ps

# Test edge proxy health endpoint
curl http://localhost/health
```

### 4. Enable Custom Domain SSL/TLS

1. Copy `nginx/conf.d/ssl.conf.template` to `nginx/conf.d/ssl.conf`.
2. Replace `drought-warning.example.org` with your registered domain name.
3. Request your initial certificate using certbot:
   ```bash
   docker compose run --rm certbot certonly --webroot -w /var/www/certbot -d yourdomain.com
   ```
4. Reload Nginx:
   ```bash
   docker compose exec reverse-proxy nginx -s reload
   ```

### 5. Stop Stack

```bash
docker compose down
```

---

## Operator Field Feedback & Ground-Truth Reconciliation API

FRADSCR incorporates a closed-loop field validation architecture connecting pastoral borehole operators and regional water bureaus across Borana Zone directly to the forecasting system:

### 1. Endpoint Specification

#### `POST /api/feedback`
Submits real-time borehole water status and ground-truth drought observations.

**Request Payload (`application/json`):**
```json
{
  "location_name": "Dubuluk Well Cluster #02",
  "latitude": 4.45,
  "longitude": 38.28,
  "observed_year": 2026,
  "observed_condition": "severe_drought",
  "borehole_yield_status": "reduced_yield",
  "water_table_depth_meters": 45.2,
  "notes": "Static water level dropped 3.2m; high pastoral livestock concentration.",
  "submitted_by": "Borana Water Bureau / Op #4"
}
```

**Field Validation & Constraints (Enforced via Zod):**
- `location_name`: string (1–100 chars, trimmed, required)
- `latitude`: float ($-90.0 \le \text{lat} \le 90.0$, required)
- `longitude`: float ($-180.0 \le \text{lon} \le 180.0$, required)
- `observed_year`: integer ($2000 \le \text{year} \le 2100$, required)
- `observed_condition`: enum (`normal_wet`, `moderate_stress`, `severe_drought`, required)
- `borehole_yield_status`: enum (`full_capacity`, `reduced_yield`, `dry_or_depleted`, required)
- `water_table_depth_meters`: float ($0.0 \le \text{depth} \le 1000.0$, optional/nullable)
- `submitted_by`: string (max 100 chars, optional)
- `notes`: string (max 500 chars, optional)

**Response:**
```json
{
  "status": "success",
  "message": "Borehole feedback logged successfully",
  "feedbackId": "6a9b36e13cd41137511ef670"
}
```

#### `GET /api/feedback`
Queries logged ground-truth observations with pagination support.

**Query Parameters:**
- `limit`: number of records (default: 50, max: 100)
- `skip`: pagination offset (default: 0)

### 2. High-Availability Low-Bandwidth Fallback
Pastoral zones in southern Ethiopia frequently experience sporadic cellular connectivity and intermittent database infrastructure. 
If MongoDB is temporarily unavailable (`mongoose.connection.readyState !== 1`):
- The feedback endpoint returns `200 OK` (`status: "accepted_ephemeral"`).
- Feedback is written directly to the structured file system audit log.
- Zero field reports are dropped or produce HTTP 500 errors for borehole operators.

---

## Satellite & Climatological Auxiliary Observation Ingestion

In addition to physical tree-ring dendrochronology and Schwabe solar cycles, FRADSCR accommodates continuous cross-validation against high-resolution Earth Observation (EO) feeds:

### 1. Satellite Observation Ingestion Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Source Validation Pipeline                    │
└─────────────────────────────────────────────────────────────────┘
         │                                       │
         ▼                                       ▼
┌───────────────────────────────┐     ┌───────────────────────────┐
│   Satellite Earth Observation │     │  Operator Ground-Truth    │
│   - CHIRPS Precipitation (0.05°)│     │  - Static Water Levels    │
│   - MODIS/Sentinel NDVI/EVI   │     │  - Pump Operational Quota │
│   - SMAP L4 Soil Moisture      │     │  - Livestock Stress Notes │
└───────────────────────────────┘     └───────────────────────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Ground-Truth Reconciliation │
              │ - Confusion Matrix Tracking │
              │ - Brier Reliability Score   │
              │ - False Alarm Mitigation    │
              └─────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Retraining & Model Updating │
              │ (Zero-Leakage Holdout Rule) │
              └─────────────────────────────┘
```

### 2. Supported Auxiliary Datasets
1. **CHIRPS v2.0 (Rainfall Anomalies)**: High-resolution ($0.05^\circ$) quasi-global precipitation estimates used to independently confirm Kiremt / Belg monsoon deficit timing.
2. **MODIS Terra/Aqua & Sentinel-2 (NDVI/NDWI)**: Normalized Difference Vegetation and Water Indices providing real-time pasture greenness and water canopy metrics.
3. **NASA SMAP L4 (Root-Zone Soil Moisture)**: $9\,\text{km}$ global soil moisture anomalies measuring deep aquifer replenishment potential.
4. **NOAA PSL Oceanic Teleconnections**: Operational updates for El Niño–Southern Oscillation (Niño 3.4 index) and Indian Ocean Dipole (Dipole Mode Index - DMI).



