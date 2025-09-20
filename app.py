import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 1. Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.joblib")
    return model

model = load_model()

# -------------------------------
# 2. Streamlit App
# -------------------------------
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

st.title("🌾 Crop Yield Prediction App")
st.write("Enter the details below to predict crop yield (hg/ha_yield).")

# -------------------------------
# 3. User Input Form
# -------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (encoded)", min_value=0, step=1)
        item = st.number_input("Item (encoded)", min_value=0, step=1)
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2020)

    with col2:
        rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, format="%.2f")
        pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, format="%.2f")
        avg_temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=60.0, format="%.2f")

    submit = st.form_submit_button("🔍 Predict")

# -------------------------------
# 4. Prediction
# -------------------------------
if submit:
    input_data = pd.DataFrame(
        [[area, item, year, rainfall, pesticides, avg_temp]],
        columns=["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"🌱 Predicted Yield: **{prediction:.2f} hg/ha**")
