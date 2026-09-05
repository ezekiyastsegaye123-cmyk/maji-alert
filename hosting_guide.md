# FRADSCR Hosting & Deployment Guide

## 1. Platform Evaluation & Recommendation

The **FRADSCR (Maji Alert)** system is architected as a **multi-runtime container** running:
1. **Node.js Express (Port 3000)**: Web UI, API gateway, rate-limiting, and client routing.
2. **Python FastAPI ML Engine (Port 8000)**: Machine learning inference (LightGBM/XGBoost/CatBoost, NetCDF, TreeRing/SPEI climate datasets).
3. **Supervisor**: Process manager keeping both services alive inside a single Docker container.

### Platform Comparison Matrix

| Platform | Best For | Docker Support | Free / Starting Tier | East Africa Latency | Recommendation Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Render** | **Production Web App + API** | Native (`render.yaml`) | Free tier available / $7/mo Starter | High (Frankfurt `fra` region) | **#1 (Top Pick)** |
| **Railway** | **Fastest Developer Setup** | Native (`railway.json`) | $5 free trial / Pay-as-you-go | Medium-High (EU / US) | **#2 (Runner-Up)** |
| **Fly.io** | **Global edge & custom scaling** | Native (`fly.toml`) | Free trial / usage-based | High (Johannesburg `jnb`, Frankfurt `fra`) | **#3** |
| **DigitalOcean** | **Fixed monthly cost VPS (Droplet)** | Full Root / Docker | $4–$6/month flat | High (Frankfurt) | **#4 (Best Budget VPS)** |
| **Hugging Face** | **Standalone ML demos (Gradio/Streamlit)** | Custom Docker Space | Generous free CPU / GPU add-ons | Medium (US/EU) | **#5** |

### Why **Render** is the Best Choice for FRADSCR:
- **Pre-configured Blueprint**: Your repository already has [`render.yaml`](./render.yaml) configured with Frankfurt region (`region: frankfurt`), custom domain mappings (`fradscr.pro.et`), and runtime variables.
- **Unified Container Support**: Handles the dual Node.js + Python stack via your [`Dockerfile`](./Dockerfile) without needing multiple separate billing services.
- **Zero-downtime Continuous Deployment**: Deploys automatically on every `git push origin main`.
- **Free Automatic SSL & DDoS Protection**: Managed TLS certificates for your custom domain.

---

## 2. Step-by-Step Deployment Plan: Render (Recommended)

### Step 1: Push Code to GitHub
Ensure all recent changes are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "chore: prepare for production deployment"
git push origin main
```

### Step 2: Sign Up & Connect Repository
1. Go to [Render Dashboard](https://dashboard.render.com/) and log in (recommended: Sign in with GitHub).
2. Click **New +** in the top right corner and select **Blueprint** (or **Web Service**).
   - If using **Blueprint**: Render detects your `render.yaml` and configures everything automatically.
   - If using **Web Service**:
     - Connect your `Fradscr` repository.
     - Environment: **Docker**.
     - Dockerfile Path: `./Dockerfile`.
     - Region: **Frankfurt (EU Central)** (ideal latency for East Africa).
     - Instance Type: **Starter** (or Free tier to test).

### Step 3: Configure Environment Variables
Navigate to the **Environment** tab in your Render service and add:
- `PORT`: `3000`
- `NODE_ENV`: `production`
- `ML_SERVICE_URL`: `http://127.0.0.1:8000`
- `ML_REQUEST_TIMEOUT_MS`: `15000`
- `CALIBRATION_TEMPERATURE`: `0.35`
- `RATE_LIMIT_WINDOW_MS`: `60000`
- `RATE_LIMIT_MAX_REQUESTS`: `60`
- `MONGODB_URI`: *(Optional)* Your MongoDB Atlas connection string (e.g. `mongodb+srv://<user>:<password>@cluster.mongodb.net/fradscr?retryWrites=true&w=majority`).

### Step 4: Deploy & Verify
1. Click **Create Web Service** / **Apply Blueprint**.
2. Monitor the build logs:
   - Stage 1 compiles Node dependencies.
   - Stage 2 installs Python scientific packages (`requirements-docker.txt`) and sets up Supervisor.
3. Test your health endpoint once deployed:
   ```bash
   curl -I https://<your-service-name>.onrender.com/health
   ```
   Expected response: `HTTP/2 200 OK` with JSON `{"status":"healthy"}`.

### Step 5: Connect Custom Domain (`fradscr.pro.et`)
1. In Render, go to **Settings** -> **Custom Domains**.
2. Add `fradscr.pro.et` and `www.fradscr.pro.et`.
3. In your domain registrar's DNS settings (e.g., Ethio Telecom / your DNS provider):
   - Add a **CNAME** record: `www` pointing to `<your-service-name>.onrender.com`.
   - Add an **ALIAS** or **ANAME** (or A-record as instructed by Render) for root `fradscr.pro.et`.
4. Render will verify the DNS records and issue a free Let's Encrypt SSL certificate automatically.

---

## 3. Alternative Quick Deploy: Railway

If you prefer **Railway**:
1. Go to [railway.app](https://railway.app) and click **New Project** -> **Deploy from GitHub repo**.
2. Select your `Fradscr` repo.
3. Railway will recognize `railway.json` and build using `Dockerfile`.
4. Add the same environment variables under the **Variables** tab.
5. In **Settings** -> **Networking**, click **Generate Domain** or link `fradscr.pro.et`.

---

## 4. Maintenance & Monitoring

- **Application Logs**: View unified supervisor logs in Render's log tab (showing both Express access logs and FastAPI inference telemetry).
- **Inference Diagnostics**: Query the test prediction endpoint:
  ```bash
  curl -X POST https://<your-service-url>/api/predict \
    -H "Content-Type: application/json" \
    -d '{"latitude": 9.03, "longitude": 38.74, "year": 2026}'
  ```
- **Alerts & Restarts**: Supervisor handles automatic subprocess restarts if Python or Node ever crashes.
