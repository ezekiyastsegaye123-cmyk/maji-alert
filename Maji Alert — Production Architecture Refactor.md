# Maji Alert — Production Architecture Refactor
## Persistent Python ML Service + Node.js HTTP Integration

### ROLE

Act as a **senior full-stack engineer, Python ML infrastructure engineer, backend architect, DevOps engineer, security engineer, and production QA engineer**.

You are refactoring the Maji Alert drought notification system because the current architecture launches:

```text
Node.js
   ↓
child_process.exec()
   ↓
python predict_service.py
   ↓
load NetCDF + Joblib + solar data
   ↓
prediction
   ↓
process exits
```

on every prediction request.

This causes a severe cold-start penalty because the heavy NetCDF grid and Joblib model are repeatedly loaded from disk, resulting in frontend `"Request timed out"` errors.

The target architecture is:

```text
Browser
   ↓
Socket.io
   ↓
Node.js / Express
   ↓ HTTP
Persistent FastAPI ML service
   ↓
Pre-loaded model/data in RAM
   ↓
Prediction
   ↓
Node.js
   ↓
Socket.io
   ↓
Browser
```

The Python process must remain alive and keep its heavy ML/scientific resources in memory.

---

# 0. NON-NEGOTIABLE RULES

Before changing anything:

1. Inspect the existing repository.
2. Inspect the current `predict_service.py`.
3. Inspect the actual Joblib model path and filename.
4. Inspect the actual SILSO sunspot dataset and loading logic.
5. Inspect the actual `spei01.nc` file and variable/coordinate structure.
6. Inspect the existing Node.js `server.js`.
7. Inspect the existing `app.js`.
8. Inspect the existing Socket.io event names and payload format.
9. Inspect existing environment variables.
10. Inspect existing tests and package versions.

### DO NOT ASSUME

Do not invent:

- model paths;
- NetCDF variable names;
- coordinate names;
- year ranges;
- feature schemas;
- prediction classes;
- API payload structures;
- existing event names;
- ports;
- environment variable names if they already exist;
- FastAPI response fields;
- Python dependency versions.

Verify them from the repository and preserve existing contracts where possible.

If an important dependency or contract is unclear, resolve it by inspecting the source/configuration rather than silently inventing behavior.

---

# 1. TARGET ARCHITECTURE

Refactor to this persistent architecture:

```text
                         ┌──────────────────────┐
                         │       Browser        │
                         │ HTML/CSS/JavaScript  │
                         └──────────┬───────────┘
                                    │
                              Socket.io
                                    │
                         ┌──────────▼───────────┐
                         │   Node.js Express    │
                         │       Backend        │
                         └──────────┬───────────┘
                                    │
                              HTTP / fetch
                                    │
                         ┌──────────▼───────────┐
                         │   FastAPI + Uvicorn  │
                         │ Persistent Python    │
                         │ ML Prediction Server │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              Joblib Model     SILSO Data      SPEI NetCDF
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                             RAM / inference
```

The Python process must be persistent.

The Node.js process must **never spawn Python for every prediction**.

---

# 2. PYTHON SERVICE — FASTAPI

Refactor `predict_service.py` into a persistent FastAPI application.

Use:

- FastAPI
- Uvicorn
- the project's existing scientific Python dependencies
- the existing ML stack

Do not rewrite the underlying ML methodology unless a verified bug makes it necessary.

The Python service remains the authoritative prediction engine.

---

# 3. GLOBAL APPLICATION STATE

Heavy resources must be loaded **once per Python worker process** during application startup.

At minimum, verify and load:

```text
random_forest_eth007.joblib
SILSO sunspot data
spei01.nc
```

into application-managed state.

Conceptually:

```python
model = None
sunspot_data = None
spei_data = None
```

or, preferably, a structured application state object.

Do not reopen the Joblib model, SILSO file, or NetCDF dataset on every `/predict` request.

Do not repeatedly instantiate expensive scientific objects inside the endpoint.

---

# 4. STARTUP LIFECYCLE

Use FastAPI's modern application lifespan/startup mechanism appropriate for the installed FastAPI version.

During startup:

1. Load the Joblib model.
2. Load or memory-map the required SILSO data.
3. Open/load the SPEI NetCDF resource.
4. Validate that every resource is usable.
5. Validate required variables exist.
6. Validate coordinates/time dimensions.
7. Validate the model can accept the expected feature structure.
8. Mark the service as ready only after all required resources have successfully initialized.

Example conceptual lifecycle:

```text
process starts
      ↓
FastAPI startup
      ↓
load model
      ↓
load solar data
      ↓
load SPEI
      ↓
validate resources
      ↓
READY
      ↓
serve /predict
```

If startup loading fails, **do not start accepting prediction requests as if the service were healthy**.

Fail clearly and log the root cause.

---

# 5. WORKER / MEMORY SAFETY

Determine how Uvicorn is being launched.

This is critical:

```text
1 worker = 1 model/data copy
2 workers = potentially 2 model/data copies
4 workers = potentially 4 model/data copies
```

Do not blindly configure multiple workers because doing so may multiply the memory footprint of the heavy scientific data.

For the default deployment, use an explicitly documented worker strategy appropriate for the available memory.

If multiple workers are recommended for production, explain the memory implications.

Do not claim the model is loaded “once globally” across multiple worker processes because Python process isolation means each worker has its own memory.

---

# 6. PYTHON API ENDPOINT

Expose:

```text
GET /predict
```

The endpoint must accept the parameters actually required by the ML pipeline.

The initial requested contract mentions:

```text
latitude
year
```

However, **inspect the existing prediction logic before finalizing the endpoint contract**.

Do not omit longitude if the scientific model or SPEI geographic lookup requires longitude.

If both are required, the endpoint must use:

```text
latitude
longitude
year
```

and validate all three.

Do not silently discard a required geographic coordinate.

---

# 7. PYTHON INPUT VALIDATION

Validate endpoint parameters using FastAPI/Pydantic.

At minimum:

```text
latitude: -90 to 90
longitude: -180 to 180
year: valid supported year
```

Reject:

- missing values;
- invalid numbers;
- NaN;
- Infinity;
- unsupported years;
- impossible coordinate values.

The validation rules must match the scientific dataset/model domain actually found in the repository.

Do not invent valid year ranges.

Return appropriate HTTP status codes.

---

# 8. PREDICTION IMPLEMENTATION

The `/predict` endpoint must perform only lightweight request-specific operations.

Conceptually:

```text
request
 ↓
validate input
 ↓
retrieve required solar value(s)
 ↓
retrieve nearest/required SPEI grid value
 ↓
construct model feature vector
 ↓
model inference
 ↓
format response
```

Do not reload:

```text
random_forest_eth007.joblib
spei01.nc
SILSO dataset
```

inside the endpoint.

Do not reconstruct expensive static data structures on every request unless measurement proves they are trivial.

Where scientifically valid, precompute reusable indexing/mappings during startup.

---

# 9. SCIENTIFIC CORRECTNESS

Do not change the scientific meaning of the existing model.

Preserve:

- existing feature engineering;
- lag structure;
- preprocessing;
- class mapping;
- scaling;
- geographical lookup logic;
- model input ordering;
- prediction semantics.

The FastAPI refactor is an **infrastructure optimization**, not an opportunity to redesign the ML methodology.

If the existing implementation has a scientific bug, document it separately rather than silently changing the model behavior.

---

# 10. MODEL FEATURE CONTRACT

The feature vector passed to:

```python
random_forest_eth007.joblib
```

must exactly match the feature schema used when the model was trained.

Verify:

- feature names;
- feature ordering;
- dtypes;
- lag features;
- standardized values;
- missing-value handling;
- year alignment;
- SPEI lookup behavior.

Do not rebuild the feature schema based on assumptions.

If the training pipeline stores a feature list or metadata artifact, use it.

---

# 11. SPEI NETCDF HANDLING

Inspect the real NetCDF file.

Verify:

- variable name;
- latitude coordinate;
- longitude coordinate;
- time coordinate;
- dimension order;
- units;
- missing values;
- geographic extent;
- temporal coverage.

Determine whether the existing pipeline uses:

- nearest-neighbor lookup;
- interpolation;
- annual aggregation;
- monthly values;
- another established method.

Preserve that verified methodology.

Do not silently change the spatial or temporal extraction method.

---

# 12. NETCDF MEMORY STRATEGY

Because the application is intended to remain persistent:

- avoid reopening the file for each request;
- avoid repeatedly parsing metadata;
- avoid unnecessary copies of the entire dataset;
- use an appropriate xarray/netCDF loading strategy;
- release unnecessary temporary objects.

Determine whether the dataset should be:

- fully loaded into RAM;
- lazily opened;
- cached;
- selectively loaded.

Choose based on the actual file size and access pattern.

Document the decision.

Do not claim the entire NetCDF grid is resident in RAM unless it actually is.

---

# 13. SILSO DATA

Inspect the existing SILSO data format and loading code.

Load the required solar data once during startup.

Build efficient lookup structures during startup where appropriate.

Do not repeatedly read CSV/text files from disk for each prediction.

Validate requested years against actual data availability.

Define a deterministic response when a year is unavailable.

---

# 14. RESPONSE CONTRACT

The endpoint must return JSON.

Preserve the existing model output contract where possible.

The response should contain the verified prediction data, including fields such as:

```json
{
  "predicted_drought_class": 0,
  "confidence_probabilities": {
    "0": 0.72,
    "1": 0.21,
    "2": 0.07
  }
}
```

But **do not invent the exact shape if the existing Python application already defines one**.

Inspect and preserve the established schema.

Validate the output before returning it.

---

# 15. PYTHON HEALTH / READINESS ENDPOINT

Create:

```text
GET /health
```

and preferably distinguish readiness from basic process liveness.

For example:

```text
GET /health
GET /ready
```

The readiness endpoint should report that:

```text
model loaded
solar data loaded
SPEI data loaded
```

only after successful startup initialization.

A service that is running but whose model failed to load must not report itself as fully ready.

Do not expose internal filesystem paths or sensitive configuration.

---

# 16. PYTHON ERROR HANDLING

The FastAPI service must handle:

- invalid coordinates;
- unsupported years;
- missing solar data;
- missing SPEI data;
- missing grid cell;
- ML inference errors;
- unexpected internal exceptions.

Return clean HTTP errors.

Do not expose:

- Python tracebacks;
- local filesystem paths;
- environment variables;
- secrets;
- internal implementation details.

Detailed technical information may be logged server-side.

---

# 17. NODE.JS REFACTOR

Remove Python process spawning from `server.js`.

Remove:

```javascript
child_process
exec
spawn
execFile
```

for ML prediction execution.

The Node.js backend must communicate with FastAPI using HTTP.

Use native Node.js `fetch()` when supported by the project's Node.js version.

Conceptually:

```text
Node.js
   ↓
fetch("http://localhost:8000/predict?...") 
   ↓
FastAPI
   ↓
JSON
```

Do not invoke the Python interpreter from Node.js for prediction requests.

---

# 18. PYTHON SERVICE URL

Do not hardcode production infrastructure details.

Use an environment variable such as:

```text
ML_SERVICE_URL
```

Example:

```text
ML_SERVICE_URL=http://127.0.0.1:8000
```

The exact name should follow existing project conventions if one already exists.

The endpoint should be constructed safely.

Do not concatenate unvalidated raw query parameters into arbitrary URLs.

Use `URL` / `URLSearchParams` or equivalent safe URL construction.

---

# 19. NODE HTTP TIMEOUT

The Node.js → FastAPI request must have an explicit timeout.

Do not allow a fetch request to hang indefinitely.

Use an appropriate `AbortController` timeout.

The value must be configurable, for example:

```text
ML_REQUEST_TIMEOUT_MS
```

Choose a sensible default based on actual observed inference latency.

Do not set an arbitrarily enormous timeout merely to hide performance issues.

---

# 20. NODE RETRY POLICY

Do not blindly retry every prediction request.

Because prediction requests may be expensive, retries can amplify load.

Implement retries only where justified, preferably for transient connectivity/startup failures.

Do not automatically repeat a successful prediction.

Document retry behavior.

---

# 21. NODE ML RESPONSE VALIDATION

Do not blindly trust the FastAPI response.

After `fetch()`:

1. Check HTTP status.
2. Parse JSON safely.
3. Validate the response structure.
4. Confirm required model fields exist.
5. Reject malformed responses.
6. Return a controlled error to the browser.

Use Zod on the Node.js side to validate the response from FastAPI where appropriate.

This creates a validation boundary:

```text
Browser → Node validation
Node → FastAPI
FastAPI → Python validation
FastAPI → Node response validation
Node → Browser
```

---

# 22. SOCKET.IO SERVER CONFIGURATION

Update Socket.io server configuration to explicitly use:

```javascript
pingTimeout: 60000
pingInterval: 25000
```

Preserve all other existing Socket.io configuration unless there is a reason to change it.

Do not treat increased Socket.io heartbeat settings as a substitute for fixing the actual ML cold-start problem.

The root performance fix is the persistent FastAPI process.

---

# 23. SOCKET.IO REQUEST FLOW

The final flow should be:

```text
Client submits coordinates
       ↓
Socket.io event
       ↓
Node validates payload
       ↓
Node requests FastAPI
       ↓
FastAPI performs in-memory inference
       ↓
FastAPI returns JSON
       ↓
Node validates response
       ↓
Node optionally stores QueryLog
       ↓
Node emits result to requesting socket
       ↓
Frontend renders result
```

Do not broadcast private prediction responses to unrelated clients.

---

# 24. FRONTEND SOCKET.IO CLIENT

Update `app.js` so that the client is compatible with:

```text
pingTimeout: 60000
pingInterval: 25000
```

Use the actual Socket.io client configuration supported by the installed client version.

Do not falsely assume the client can configure server heartbeat behavior independently.

The important requirement is that the client connection remains resilient to legitimate periods of activity while the server performs inference.

Do not introduce unnecessary reconnection loops.

---

# 25. FRONTEND LOADING STATE

Immediately after the user submits valid coordinates, update the UI.

Display a clear loading state such as:

```text
Analyzing Regional Climate Data...
```

Use the project's localization dictionary rather than hardcoding English into a multilingual interface.

The loading state must:

- appear immediately;
- disable duplicate submission where appropriate;
- prevent users from assuming the page froze;
- remain visible until success or failure;
- disappear on timeout/error;
- provide a retry mechanism.

Do not leave the UI permanently disabled after a failed request.

---

# 26. LOCALIZATION

Preserve support for:

```text
English
Amharic
Afaan Oromoo
```

Add the loading-state translation to all supported languages.

Also ensure prediction errors and connection errors are localized.

Do not scatter strings throughout the codebase.

---

# 27. FRONTEND TIMEOUT / RESILIENCY

The browser must have a bounded waiting state.

If a prediction does not return within the expected time:

```text
loading
   ↓
timeout
   ↓
user-friendly error
   ↓
retry option
```

Do not silently fail.

Do not tell the user the model is working when the connection has already failed.

---

# 28. DOUBLE-SUBMISSION PROTECTION

Prevent accidental rapid duplicate predictions.

Possible behavior:

```text
submit
 ↓
button disabled
 ↓
request active
 ↓
result/error
 ↓
button enabled
```

Do not create multiple simultaneous expensive ML requests from a single user action.

---

# 29. BACKEND HEALTH-AWARE ERROR MESSAGES

If Node.js cannot reach FastAPI, distinguish that from an invalid coordinate request.

For example:

```text
Invalid coordinates
```

versus:

```text
Prediction service unavailable
```

versus:

```text
Prediction request timed out
```

Use safe, user-friendly localized messages.

Do not expose raw `ECONNREFUSED`, stack traces, or Axios/fetch internals.

---

# 30. DATABASE LOGGING

Preserve the existing `QueryLog` MongoDB behavior.

When a prediction succeeds:

```text
coordinates
timestamp
prediction
```

and any other already-required fields should be stored according to the existing schema.

Do not allow a MongoDB outage to crash the prediction service.

Determine whether database persistence should:

- block the response;
- happen asynchronously;
- or fail independently.

Preserve the existing application semantics unless there is a clear reliability reason to change them.

Document the decision.

---

# 31. SECURITY

The refactor must preserve all existing security controls.

Verify:

- Zod request validation;
- CORS configuration;
- security headers;
- rate limiting;
- request-size limits;
- safe URL construction;
- no shell execution;
- no arbitrary command execution;
- no sensitive error leakage;
- no secrets in source code.

The removal of `child_process` should eliminate the command-injection attack surface associated with passing coordinates to a shell.

---

# 32. CONCURRENCY

Determine whether the Python model can safely process multiple concurrent inference requests.

If the model/object state is read-only during prediction, verify that assumption.

If inference or scientific data access is not concurrency-safe:

- implement appropriate synchronization;
- or use an appropriate execution model;
- or explicitly limit concurrent predictions.

Do not assume thread safety without checking the libraries and actual code.

---

# 33. PERFORMANCE VERIFICATION

The purpose of this refactor is to eliminate repeated cold starts.

Measure before and after:

### Old architecture

```text
request
→ Python startup
→ model/data loading
→ inference
→ process exit
```

### New architecture

```text
FastAPI already running
→ request
→ in-memory inference
```

Measure:

- first startup time;
- model/data initialization time;
- typical prediction latency;
- P95 prediction latency if enough test samples exist;
- Node → FastAPI HTTP latency;
- total browser request latency.

Do not claim the timeout problem is solved without testing actual latency.

---

# 34. COLD-START VERIFICATION TEST

Create a test or diagnostic procedure that proves the heavy resources are not reloaded on every request.

For example, instrument startup loading so logs clearly show:

```text
Loading Joblib model...
Loading SILSO...
Loading SPEI...
ML service ready.
```

These messages should occur during process startup rather than once per prediction.

Then issue several prediction requests and verify that the heavy-loading messages are not repeated.

Do not rely solely on visual inspection.

---

# 35. MEMORY VERIFICATION

Measure process memory after startup.

Document approximate memory usage if practical.

Pay particular attention to:

```text
spei01.nc
Joblib model
Python scientific libraries
```

Do not introduce multi-worker configuration that unintentionally multiplies memory consumption without documenting it.

---

# 36. STARTUP FAILURE TEST

Test scenarios where:

- Joblib model file is missing;
- SILSO file is missing;
- NetCDF file is missing;
- NetCDF variable is wrong;
- model is incompatible;
- dataset cannot be opened.

The FastAPI service must fail clearly rather than entering a deceptive partially-ready state.

---

# 37. FASTAPI API TESTS

Test at minimum:

### `/predict`

- valid latitude;
- valid longitude;
- valid year;
- invalid latitude;
- invalid longitude;
- invalid year;
- missing parameters;
- unavailable year;
- unavailable geographical data;
- malformed internal model output.

### `/health`

- service running.

### `/ready`

- service fully initialized;
- service not ready before initialization completes.

---

# 38. NODE INTEGRATION TESTS

Test:

- Node can contact FastAPI;
- valid prediction response;
- invalid FastAPI response;
- FastAPI 4xx response;
- FastAPI 5xx response;
- connection refused;
- ML timeout;
- malformed JSON;
- retry behavior;
- Socket.io result delivery.

---

# 39. FRONTEND TESTS

Test:

- prediction submission;
- immediate loading state;
- successful result;
- timeout;
- FastAPI unavailable;
- Socket.io disconnect;
- Socket.io reconnect;
- duplicate submission;
- geolocation success;
- manual coordinate fallback;
- English;
- Amharic;
- Afaan Oromoo.

---

# 40. PRODUCTION PROCESS MANAGEMENT

Document how FastAPI should run in production.

For example:

```text
uvicorn predict_service:app --host 127.0.0.1 --port 8000
```

But **verify the actual module/app names before documenting the command**.

Do not expose port 8000 publicly unless the deployment architecture explicitly requires it.

Prefer Node.js communicating with the Python service through localhost/private networking.

For production deployment, document an appropriate process manager/container/supervisor strategy.

Do not leave the Python process dependent on a developer terminal session.

---

# 41. DEVELOPMENT COMMANDS

Update documentation with the actual verified commands for:

```text
install dependencies
start FastAPI
start Node.js
run tests
```

Do not provide commands that were not validated against the project.

---

# 42. ENVIRONMENT VARIABLES

Use environment variables for configuration.

At minimum consider:

```text
ML_SERVICE_URL
ML_REQUEST_TIMEOUT_MS
PORT
MONGODB_URI
NODE_ENV
CORS_ORIGIN
```

The exact names should follow existing project conventions where possible.

Create/update:

```text
.env.example
```

Never commit production secrets.

---

# 43. CODE QUALITY

The final implementation must have:

- separation of concerns;
- clear module boundaries;
- reusable configuration;
- centralized validation;
- consistent error handling;
- no duplicated API logic;
- no dead code;
- no debug statements;
- meaningful names;
- useful comments only where they explain non-obvious technical decisions.

Do not create unnecessary abstractions.

---

# 44. EXPECTED PROJECT STRUCTURE

Adapt to the existing repository.

A reasonable target may look like:

```text
project/
├── public/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── src/
│   ├── config/
│   ├── models/
│   ├── services/
│   ├── validation/
│   └── utils/
│
├── tests/
│
├── predict_service.py
├── server.js
├── package.json
├── .env.example
├── .gitignore
└── README.md
```

Do not reorganize the repository unnecessarily if the existing structure is already production-appropriate.

---

# 45. REQUIRED IMPLEMENTATION OUTPUT

Actually implement the refactor in the repository.

At minimum, modify the verified relevant files:

```text
predict_service.py
server.js
app.js
```

and create/update supporting files only where required.

Potential supporting files include:

```text
Python requirements
.env.example
tests
FastAPI service modules
Node service modules
README.md
```

Do not fabricate files that are irrelevant to the existing architecture.

---

# 46. REQUIRED FINAL REPORT

After implementation, provide:

## A. Architecture Before

Explain the original cold-start architecture.

## B. Architecture After

Explain:

```text
Browser
→ Socket.io
→ Node.js
→ HTTP fetch
→ persistent FastAPI
→ in-memory ML/data
→ response
→ browser
```

## C. Heavy Resource Lifecycle

Explicitly document:

```text
Joblib model: loaded when?
SILSO data: loaded when?
SPEI NetCDF: loaded when?
```

and demonstrate that they are not reloaded per request.

## D. API Contract

Document the actual verified:

```text
GET /predict
GET /health
GET /ready
```

request and response schemas.

## E. Socket.io Configuration

Document:

```text
pingTimeout = 60000
pingInterval = 25000
```

and the reasoning.

## F. Performance

Report measured:

- startup time;
- first prediction;
- subsequent prediction latency;
- Node → Python latency;
- relevant memory usage.

Do not invent numbers.

## G. Testing

Report:

- tests run;
- tests passed;
- tests failed;
- cold-start verification;
- timeout verification;
- startup failure tests;
- concurrency tests;
- security tests.

## H. Known Limitations

List every unresolved issue.

## I. Production Readiness

Return exactly one:

```text
PRODUCTION READY
```

or:

```text
NOT PRODUCTION READY
```

If `NOT PRODUCTION READY`, list the blocking issues.

---

# 47. FINAL PRODUCTION GATE

Do not declare success merely because:

```text
uvicorn starts
```

or:

```text
Node.js starts
```

The refactor is complete only when all of the following are demonstrated:

- [ ] FastAPI starts successfully.
- [ ] Model loads once during startup.
- [ ] SILSO data loads once during startup.
- [ ] SPEI data loads once during startup.
- [ ] `/ready` becomes healthy only after initialization.
- [ ] `/predict` performs inference without reloading heavy data.
- [ ] Node.js uses HTTP instead of `child_process`.
- [ ] No shell command execution remains for ML prediction.
- [ ] FastAPI responses are validated.
- [ ] Socket.io uses `pingTimeout: 60000`.
- [ ] Socket.io uses `pingInterval: 25000`.
- [ ] Frontend displays the loading state immediately.
- [ ] Frontend recovers from timeout/failure.
- [ ] Duplicate submissions are controlled.
- [ ] Localization remains functional.
- [ ] MongoDB logging remains functional.
- [ ] Security controls remain intact.
- [ ] Automated/integration tests pass.
- [ ] Measured latency demonstrates improvement.
- [ ] Memory impact of the persistent service is understood.
- [ ] Deployment/startup instructions are documented.

---

# FINAL INSTRUCTION

This is a **production architecture refactor**, not a superficial timeout increase.

Do not solve the problem by simply increasing frontend or Socket.io timeouts.

The primary objective is:

```text
ELIMINATE PER-REQUEST PYTHON COLD START
```

by maintaining a persistent FastAPI process with verified, pre-loaded ML/scientific resources.

Do not hide errors.

Do not fabricate successful tests.

Do not claim that resources are loaded once unless the implementation and runtime evidence demonstrate it.

Do not change the scientific model's behavior merely to make the application appear faster.

Preserve scientific correctness while improving the system's runtime architecture, reliability, security, and observability.