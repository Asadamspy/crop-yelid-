import streamlit as st
import pandas as pd
import joblib

# --- Page config ---
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Yield Prediction App")
st.write("Predict the crop yield based on fertilizer, rainfall, and temperature.")

# --- Load model ---
@st.cache_resource
def load_model():
    # Make sure your model file is in the same folder
    model = joblib.load("random_forest_model.joblib")
    return model

model = load_model()

# --- User input form ---
with st.form("prediction_form"):
    st.subheader("Enter input parameters:")
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=50.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, value=25.0)
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Create input DataFrame
        input_df = pd.DataFrame([[fertilizer, rainfall, temperature]],
                                columns=["fertilizer", "rainfall", "temperature"])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        st.success(f"Predicted Crop Yield: {prediction:.2f} tons/ha")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Your Name | Powered by Streamlit & Hugging Face Spaces")
