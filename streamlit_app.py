"""
FRADSCR — Solar Groundwater Pump Drought Early Warning System
=============================================================
Streamlit Cloud authoritative deployment interface.
Powered by:
- Tree-ring growth memory (RWI)
- 11-Year Schwabe solar cycle teleconnections (SILSO)
- Oceanic dipole anomalies (ENSO / IOD)
- High-resolution spatial SPEI grids
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict_service import predict_drought, SEVERITY_LABELS

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="FRADSCR — Drought Early Warning",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-contrast, clean UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0284c7;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(2, 132, 199, 0.05);
        border: 1px solid rgba(2, 132, 199, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-normal {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        color: #065f46;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    .alert-moderate {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    .alert-severe {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Presets & Locations
# =============================================================================
PRESETS = {
    "Borana — Yabelo (4.88° N, 38.08° E)": (4.88, 38.08),
    "Borana — Dubuluk (4.45° N, 38.28° E)": (4.45, 38.28),
    "Borana — Mega (4.05° N, 38.32° E)": (4.05, 38.32),
    "Borana — Moyale (3.53° N, 39.05° E)": (3.53, 39.05),
    "Gondar / Highlands (12.60° N, 37.47° E)": (12.60, 37.47),
    "Debrebirkan Selassie (9.63° N, 39.53° E)": (9.63, 39.53),
    "Somali Region — Gode (5.95° N, 43.55° E)": (5.95, 43.55),
    "Afar Region — Semera (11.79° N, 41.01° E)": (11.79, 41.01),
    "Custom Coordinates": None
}

# =============================================================================
# Sidebar Controls
# =============================================================================
st.sidebar.image("https://raw.githubusercontent.com/ezekiyastsegaye123-cmyk/maji-alert/main/public/favicon.svg" if False else "💧", width=48)
st.sidebar.title("Telemetry Controls")

selected_preset = st.sidebar.selectbox("Location Preset", list(PRESETS.keys()))

if PRESETS[selected_preset] is not None:
    default_lat, default_lon = PRESETS[selected_preset]
else:
    default_lat, default_lon = 4.88, 38.08

latitude = st.sidebar.number_input("Latitude (°N/S)", min_value=-90.0, max_value=90.0, value=default_lat, step=0.01, format="%.2f")
longitude = st.sidebar.number_input("Longitude (°E/W)", min_value=-180.0, max_value=180.0, value=default_lon, step=0.01, format="%.2f")
target_year = st.sidebar.slider("Forecast Year", min_value=1900, max_value=2035, value=2026, step=1)

with st.sidebar.expander("Model Hyperparameters & Calibration"):
    calib_temp = st.slider("Temperature Scaling (T)", min_value=0.10, max_value=2.00, value=0.35, step=0.05,
                           help="Optimal temperature scaling parameter for probability calibration (T=0.35).")

# =============================================================================
# Main Header
# =============================================================================
st.markdown("<div class=\"main-title\">FRADSCR · Drought Early Warning System</div>", unsafe_allow_html=True)
st.markdown("<div class=\"sub-title\">Solar Groundwater Pumping & Climate Teleconnection Forecasting · Borana Zone & Horn of Africa</div>", unsafe_allow_html=True)

# Run Prediction
with st.spinner("Running in-memory climate teleconnection inference..."):
    result = predict_drought(
        latitude=latitude,
        longitude=longitude,
        year=target_year,
        temperature=calib_temp
    )

cls = result["predicted_drought_class"]
severity = result["severity_label"]
probs = result["confidence_probabilities"]
p0 = probs.get("class_0", 0.0) * 100
p1 = probs.get("class_1", 0.0) * 100
p2 = probs.get("class_2", 0.0) * 100
confidence = result["model_confidence"] * 100
combined_risk = result["combined_drought_risk"] * 100
tier = result.get("drought_risk_tier", "Guarded Risk")

# =============================================================================
# Primary Risk Alert Banner & Key Metrics
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Evaluation Year", f"{target_year}")
with col2:
    st.metric("Predicted Severity", f"{severity}")
with col3:
    st.metric("Combined Drought Risk", f"{combined_risk:.1f}%", delta=tier, delta_color="inverse" if combined_risk >= 50 else "normal")
with col4:
    st.metric("Model Confidence", f"{confidence:.1f}%", help="Confidence of the leading predicted class under temperature scaling.")

if cls == 2 or (cls == 0 and combined_risk >= 60):
    st.markdown(
        f"""
        <div class="alert-severe">
            🚨 <strong>CRITICAL DROUGHT WARNING ({target_year}):</strong> Severe water deficit projected.
            Enforce emergency water conservation, prioritize human consumption, and restrict non-essential solar borehole discharge.
        </div>
        """,
        unsafe_allow_html=True
    )
elif cls == 1 or (cls == 0 and combined_risk >= 45):
    st.markdown(
        f"""
        <div class="alert-moderate">
            ⚠️ <strong>MODERATE WATER STRESS ADVISORY ({target_year}):</strong> Groundwater recharge is below decadal average.
            Implement rotational solar pumping schedules and monitor reservoir static water levels.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class="alert-normal">
            ✅ <strong>NORMAL / RECHARGE PHASE ({target_year}):</strong> Adequate groundwater recharge expected.
            Solar water pumping can operate at full standard allocation.
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# =============================================================================
# Visual Gauge & Probability Distribution
# =============================================================================
viz_col1, viz_col2 = st.columns([1, 1])

with viz_col1:
    st.subheader("Drought Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=combined_risk,
        title={'text': f"Total Drought Risk ({target_year})", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0284c7"},
            'steps': [
                {'range': [0, 35], 'color': "#ecfdf5"},
                {'range': [35, 50], 'color': "#fef3c7"},
                {'range': [50, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': 50.0
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with viz_col2:
    st.subheader("Calibrated Class Probabilities")
    df_probs = pd.DataFrame({
        "Condition": ["Normal / Wet", "Moderate Drought", "Severe Drought"],
        "Probability (%)": [p0, p1, p2],
        "Color": ["#10b981", "#f59e0b", "#ef4444"]
    })
    fig_bar = px.bar(
        df_probs,
        x="Probability (%)",
        y="Condition",
        orientation="h",
        color="Condition",
        color_discrete_map={
            "Normal / Wet": "#10b981",
            "Moderate Drought": "#f59e0b",
            "Severe Drought": "#ef4444"
        },
        text=df_probs["Probability (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_bar.update_layout(
        height=280,
        showlegend=False,
        xaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# Decadal Forecast Trajectory (2025 – 2035)
# =============================================================================
st.subheader("Solar-Cycle Decadal Trajectory (2025–2035)")
st.caption("Forward projection integrating Schwabe Cycle 25/26 solar irradiance and tree-ring biological memory.")

years_range = list(range(2025, 2036))
traj_records = []
for y in years_range:
    r = predict_drought(latitude=latitude, longitude=longitude, year=y, temperature=calib_temp)
    p = r["confidence_probabilities"]
    traj_records.append({
        "Year": y,
        "Normal / Wet (%)": round(p.get("class_0", 0) * 100, 1),
        "Moderate Drought (%)": round(p.get("class_1", 0) * 100, 1),
        "Severe Drought (%)": round(p.get("class_2", 0) * 100, 1),
        "Combined Risk (%)": round(r["combined_drought_risk"] * 100, 1),
        "Severity": r["severity_label"]
    })

df_traj = pd.DataFrame(traj_records)

fig_traj = go.Figure()
fig_traj.add_trace(go.Scatter(x=df_traj["Year"], y=df_traj["Combined Risk (%)"], name="Total Drought Risk", line=dict(color="#0284c7", width=3)))
fig_traj.add_trace(go.Scatter(x=df_traj["Year"], y=df_traj["Severe Drought (%)"], name="Severe Drought Prob", line=dict(color="#ef4444", width=2, dash="dot")))
fig_traj.add_trace(go.Scatter(x=df_traj["Year"], y=df_traj["Moderate Drought (%)"], name="Moderate Drought Prob", line=dict(color="#f59e0b", width=2, dash="dash")))

fig_traj.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="High Risk Threshold (50%)")
fig_traj.update_layout(
    xaxis=dict(tickmode="linear", dtick=1),
    yaxis=dict(title="Probability (%)", range=[0, 100]),
    height=340,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_traj, use_container_width=True)

# =============================================================================
# Technical Telemetry Metadata
# =============================================================================
with st.expander("Technical Model & Grid Metadata"):
    grid = result.get("grid_cell", {})
    meta_c1, meta_c2, meta_c3 = st.columns(3)
    with meta_c1:
        st.write(f"**Requested Coordinates:** `{latitude:.2f}° N, {longitude:.2f}° E`")
        st.write(f"**Matched SPEI Grid Cell:** `{grid.get('selected_lat', '—')}° N, {grid.get('selected_lon', '—')}° E`")
    with meta_c2:
        st.write(f"**Grid Distance:** `{grid.get('distance_km', 0):.2f} km`")
        st.write(f"**Operational Accuracy:** `85.85%`")
    with meta_c3:
        st.write(f"**Engine Mode:** `{result.get('service_mode', 'prospective_solar_projection')}`")
        st.write(f"**Calibration Temperature:** `T = {calib_temp}`")
