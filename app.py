
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ============================================================
# TRAIN MODEL ON STARTUP (avoids pickle compatibility issues)
# ============================================================
@st.cache_resource
def train_model():
    agg = pd.read_csv('crime_risk_aggregated.csv')
    
    features = ['lat_bin', 'lon_bin', 'hour', 'day_of_week', 'is_weekend']
    X = agg[features]
    y = agg['risk_level']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    
    pipeline.fit(X, y_encoded)
    return pipeline, le, features

pipeline, le, features = train_model()

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
                     help="South Chicago ~ 41.65, North ~ 42.02")
    hour = st.slider("Hour of Day", 0, 23, 14,
                      help="0 = midnight, 12 = noon, 23 = 11 PM")

with col2:
    lon = st.slider("Longitude", -87.93, -87.52, -87.65, 0.01,
                     help="West Chicago ~ -87.93, East ~ -87.52")
    day_of_week = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                'Thursday','Friday','Saturday','Sunday'][x],
        index=5
    )

# ============================================================
# PREDICTION
# ============================================================
if st.button("Predict Risk Level", type="primary"):
    
    lat_bin = round(lat, 2)
    lon_bin = round(lon, 2)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    input_df = pd.DataFrame({
        'lat_bin': [lat_bin],
        'lon_bin': [lon_bin],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend]
    })
    
    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    pred_proba = pipeline.predict_proba(input_df)[0]
    
    st.markdown("---")
    
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    emoji_map = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    
    st.markdown(
        f"### Predicted Risk: {emoji_map[pred_label]} "
        f"<span style='color:{color_map[pred_label]}'>{pred_label}</span>",
        unsafe_allow_html=True
    )
    
    st.write("**Confidence Breakdown:**")
    prob_df = pd.DataFrame({
        'Risk Level': le.classes_,
        'Probability': pred_proba
    }).sort_values('Probability', ascending=False)
    
    for _, row in prob_df.iterrows():
        level = row['Risk Level']
        prob = row['Probability']
        st.progress(prob, text=f"{level}: {prob:.1%}")
    
    st.markdown("---")
    day_name = ['Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday'][day_of_week]
    st.caption(
        f"Prediction for: ({lat_bin}, {lon_bin}) on {day_name} at {hour}:00"
    )
