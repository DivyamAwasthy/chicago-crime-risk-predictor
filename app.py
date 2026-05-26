
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# LOAD MODEL (runs once when app starts)
# ============================================================
@st.cache_resource
def load_model():
    with open('crime_risk_model.pkl', 'rb') as f:
        package = pickle.load(f)
    return package

model_pkg = load_model()
pipeline = model_pkg['pipeline']
le = model_pkg['label_encoder']
features = model_pkg['features']

# ============================================================
# APP LAYOUT
# ============================================================
st.set_page_config(page_title="Chicago Crime Risk Predictor", layout="centered")
st.title("Chicago Crime Risk Predictor")
st.write("Predict crime risk level for any location and time in Chicago")

# ============================================================
# USER INPUTS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    lat = st.slider("Latitude", 41.65, 42.02, 41.85, 0.01,
                     help="South Chicago ≈ 41.65, North ≈ 42.02")
    hour = st.slider("Hour of Day", 0, 23, 14,
                      help="0 = midnight, 12 = noon, 23 = 11 PM")

with col2:
    lon = st.slider("Longitude", -87.93, -87.52, -87.65, 0.01,
                     help="West Chicago ≈ -87.93, East ≈ -87.52")
    day_of_week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                'Thursday','Friday','Saturday','Sunday'][x],
        index=5  # default to Saturday
    )

# ============================================================
# PREDICTION
# ============================================================
if st.button("Predict Risk Level", type="primary"):
    
    # Round lat/lon to match training grid (2 decimal places)
    lat_bin = round(lat, 2)
    lon_bin = round(lon, 2)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Build input dataframe with same columns as training
    input_df = pd.DataFrame({
        'lat_bin': [lat_bin],
        'lon_bin': [lon_bin],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend]
    })
    
    # Predict
    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    # Get prediction probabilities
    pred_proba = pipeline.predict_proba(input_df)[0]
    
    # Display result with color coding
    st.markdown("---")
    
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    emoji_map = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    
    st.markdown(
        f"### Predicted Risk: {emoji_map[pred_label]} "
        f"<span style='color:{color_map[pred_label]}'>{pred_label}</span>",
        unsafe_allow_html=True
    )
    
    # Show confidence breakdown
    st.write("**Confidence Breakdown:**")
    prob_df = pd.DataFrame({
        'Risk Level': le.classes_,
        'Probability': pred_proba
    }).sort_values('Probability', ascending=False)
    
    for _, row in prob_df.iterrows():
        level = row['Risk Level']
        prob = row['Probability']
        st.progress(prob, text=f"{level}: {prob:.1%}")
    
    # Show what inputs were used
    st.markdown("---")
    day_name = ['Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday'][day_of_week]
    st.caption(
        f"Prediction for: ({lat_bin}, {lon_bin}) on {day_name} at {hour}:00"
    )
