"""
Advanced Streamlit app for Real Estate Price Prediction
Features:
- Tailwind-like clean UI (via custom CSS)
- Location autocomplete (from dataset localities)
- Map-based input: click map to pick lat/lon
- Trend charts: median price by locality, scatter plots, correlation heatmap
- SHAP explainability panel for per-prediction breakdown
- Comparables search (nearest N)

Usage:
- Place this file in the same folder as models/house_price_model.joblib and data/sample_houses.csv and data/locality_coords.json
- Install requirements from the provided requirements_advanced.txt (below)
- Run: streamlit run streamlit_advanced_app.py

Note: SHAP plotting may require additional JS support; the code attempts to render SHAP plots with matplotlib or HTML fallback.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from streamlit_folium import st_folium
import folium
import json
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
import math

# -----------------------------
# Helpers
# -----------------------------

def haversine(lat1, lon1, lat2, lon2):
    # returns distance in km
    R = 6371.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# -----------------------------
# Layout & CSS (Tailwind-like look)
# -----------------------------
st.set_page_config(page_title="Real Estate — Advanced", layout="wide")

st.markdown("""
<style>
/* Simple Tailwind-like box styles */
:root{--card-bg:#ffffff;--muted:#6b7280;--accent:#0ea5a4}
body { background-color: #f8fafc; }
.card { background: var(--card-bg); border-radius: 12px; padding: 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
.hstack{ display:flex; gap:16px; align-items:center; }
.vstack{ display:flex; flex-direction:column; gap:12px; }
.small{ font-size:0.9rem; color:var(--muted); }
.kpi { font-weight:700; font-size:1.4rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='card'><div class='hstack'><div style='flex:1'><h2>Real Estate Price Predictor — Advanced</h2><div class='small'>Map input, charts, SHAP explainability, and cleaner UI</div></div></div></div>", unsafe_allow_html=True)

# -----------------------------
# Load data and model
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_houses.csv"
COORDS_PATH = ROOT / "data" / "locality_coords.json"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    coords = {}
    try:
        with open(COORDS_PATH, "r") as f:
            coords = json.load(f)
    except Exception:
        pass
    return df, coords

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df, locality_coords = load_data()
except FileNotFoundError:
    st.error("Sample dataset not found. Ensure data/sample_houses.csv exists.")
    st.stop()

if not MODEL_PATH.exists():
    st.warning("Model not found. Run train.py to create models/house_price_model.joblib")
    st.stop()

model = load_model()

# Prepare locality list for autocomplete
localities = sorted(df['locality'].unique().tolist())

# -----------------------------
# Sidebar — Inputs
# -----------------------------
with st.sidebar:
    st.markdown("<div class='card'><h3>Property details</h3></div>", unsafe_allow_html=True)
    area = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=900, step=50)
    bedrooms = st.select_slider("Bedrooms", options=[1,2,3,4,5], value=2)
    bathrooms = st.select_slider("Bathrooms", options=[1,2,3,4,5], value=2)
    year_built = st.slider("Year built", min_value=1980, max_value=2025, value=2015)

    # Autocomplete: searchable selectbox
    locality_input = st.selectbox("Locality (autocomplete)", options=[""] + localities, index=0)
    manual_locality = st.text_input("Or type a custom locality (won't have trend data)", value="")
    if manual_locality.strip():
        locality_input = manual_locality.strip()

    furnishing = st.selectbox("Furnishing", ["Unfurnished","Semi-furnished","Furnished"], index=0)
    property_type = st.selectbox("Property type", ["Apartment","Independent House","Villa"], index=0)

    st.markdown("---")
    st.markdown("<div class='small'>Pick a location by clicking on the map (bottom-left). Click 'Use map click' after clicking.</div>", unsafe_allow_html=True)
    use_map_click = st.checkbox("Use map click lat/lon (recommended)", value=True)
    predict_button = st.button("Predict price", key="predict")

# -----------------------------
# Main layout: left column map + comparables, right column results + SHAP + charts
# -----------------------------
left_col, right_col = st.columns([1.1, 1])

# LEFT: Map and comparables
with left_col:
    st.markdown("<div class='card'><h4>Map - click to pick location</h4></div>", unsafe_allow_html=True)
    # Center map at city center fallback
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=13)

    # Show sample points (clustered)
    from folium.plugins import MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)
    for _, r in df.sample(min(len(df), 150), random_state=42).iterrows():
        folium.CircleMarker(location=[r['latitude'], r['longitude']], radius=3, fill=True,
                            popup=f"{r['locality']} — ₹{int(r['price']):,}").add_to(marker_cluster)

    st_map = st_folium(m, width=700, height=450)

    # map click lat/lon
    clicked = None
    if st_map and st_map.get('last_clicked'):
        clicked = st_map['last_clicked']
        st.info(f"Map clicked at: {clicked['lat']:.6f}, {clicked['lng']:.6f}")

    if use_map_click and clicked:
        lat, lon = clicked['lat'], clicked['lng']
    else:
        # fallback to locality coords if available
        if locality_input and locality_input in locality_coords:
            lat, lon = locality_coords[locality_input]
        else:
            lat, lon = center[0], center[1]

    st.markdown("<div class='card'><h4>Nearest comparables</h4></div>", unsafe_allow_html=True)
    # Compute nearest 5 comparables using haversine
    df['dist_km'] = df.apply(lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)
    comps = df.nsmallest(8, 'dist_km')[['area_sqft','bedrooms','bathrooms','locality','price','dist_km']]
    st.dataframe(comps.reset_index(drop=True))

# RIGHT: Prediction, SHAP, Trend charts
with right_col:
    st.markdown("<div class='card'><h4>Prediction & Explainability</h4></div>", unsafe_allow_html=True)

    # Build input row
    input_row = pd.DataFrame([{
        'area_sqft': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'age': 2025 - year_built,
        'locality': locality_input if locality_input else (manual_locality or localities[0]),
        'furnishing': furnishing,
        'property_type': property_type,
        'latitude': lat,
        'longitude': lon
    }])

    st.write("**Input**")
    st.table(input_row.T)

    if predict_button:
        pred = model.predict(input_row.drop(columns=['year_built']))[0]
        st.metric(label="Predicted price (INR)", value=f"₹{int(pred):,}")

        # SHAP explainability
        st.markdown("<div style='margin-top:12px'><strong>SHAP explainability</strong></div>", unsafe_allow_html=True)
        try:
            # Extract preprocessor and xgb model from pipeline
            preproc = model.named_steps.get('preproc')
            xgb = model.named_steps.get('xgb')

            # prepare background small sample
            background = df.sample(min(200, len(df)), random_state=42)
            X_bg = background[['area_sqft','bedrooms','bathrooms','age','locality','furnishing','property_type','latitude','longitude']]
            X_bg_trans = preproc.transform(X_bg)
            X_input_trans = preproc.transform(input_row[['area_sqft','bedrooms','bathrooms','age','locality','furnishing','property_type','latitude','longitude']])

            # Use shap.Explainer with model; will work for Tree-based models
            explainer = shap.Explainer(xgb)
            shap_values = explainer(X_input_trans)

            # Waterfall or bar: attempt matplotlib waterfall
            try:
                fig = shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(bbox_inches='tight')
            except Exception:
                # fallback to bar (feature importance style)
                fig2 = shap.plots.bar(shap_values, show=False)
                st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.write("SHAP failed:", e)
            st.write("If SHAP fails, make sure shap is installed and your model pipeline exposes named_steps 'preproc' and 'xgb'.")

    # Trend charts
    st.markdown("<div class='card' style='margin-top:16px'><h4>Trend charts</h4></div>", unsafe_allow_html=True)
    with st.expander("Median price by locality"):
        med = df.groupby('locality')['price'].median().reset_index().sort_values('price', ascending=False)
        fig = px.bar(med, x='locality', y='price', title='Median price by locality', labels={'price':'Median price (INR)'})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Scatter plots"):
        # price vs sqft
        fig2 = px.scatter(df.sample(min(2000, len(df))), x='area_sqft', y='price', color='locality', size='bedrooms', title='Price vs Area (sample)')
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Correlation heatmap"):
        num = df[['price','area_sqft','bedrooms','bathrooms','age']]
        corr = num.corr()
        fig3 = px.imshow(corr, text_auto=True, title='Correlation matrix (numeric features)')
        st.plotly_chart(fig3, use_container_width=True)

# Footer notes
st.markdown("<div style='margin-top:18px' class='small'>Advanced features: click map to pick exact coordinates, use autocomplete locality, view SHAP breakdown and trend charts. Replace the synthetic dataset with your real dataset for production.</div>", unsafe_allow_html=True)
