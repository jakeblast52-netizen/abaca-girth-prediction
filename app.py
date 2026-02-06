import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Abaca Girth Prediction System",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
with open("abaca_rf_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
feature_cols = model_package["features"]

# ===============================
# TITLE
# ===============================
st.title("🌱 Abaca Plant Girth Prediction")
st.write(
    """
    This web-based system predicts **Abaca (Musa textilis) plant girth**
    using a **Random Forest Regression model**.
    """
)

# ===============================
# USER INPUTS
# ===============================
st.header("Input Parameters")

height_cm = st.number_input("Plant Height (cm)", 50.0, 500.0, 200.0)
leaf_count = st.number_input("Leaf Count", 1.0, 20.0, 5.0)
moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 60.0)
soil_pH = st.number_input("Soil pH", 3.0, 9.0, 6.5)
temperature = st.number_input("Temperature (°C)", 10.0, 45.0, 28.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
sun_shade = st.number_input("Sun Shade (%)", 0.0, 100.0, 60.0)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Girth"):
    input_data = pd.DataFrame([[
        height_cm,
        leaf_count,
        moisture,
        soil_pH,
        temperature,
        humidity,
        sun_shade
    ]], columns=feature_cols)

    predicted_log = model.predict(input_data)[0]
    predicted_girth = np.expm1(predicted_log)

    predicted_girth = max(predicted_girth, 0.5)

    st.success(f"🌿 Predicted Abaca Girth: **{predicted_girth:.2f} cm**")
