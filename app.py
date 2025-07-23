import streamlit as st
import pandas as pd
import joblib
from scipy.stats import rice

# Load ML model and label encoder
model = joblib.load("voltage_classifier_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load dataset to calculate mean & std
data = pd.read_csv("Project_Dataset_clean.csv")
stable_voltages = data[data["Label"] == "Stable"]["Voltage"]
kappa = stable_voltages.mean()
sigma = stable_voltages.std()

# Function to calculate Pv(t)
def calculate_pv(voltage, kappa, sigma):
    if voltage == 0:
        return 1.0
    b = kappa / sigma
    lower_v, upper_v = 218.5, 241.5
    prob_stable = rice.cdf(upper_v, b=b, scale=sigma) - rice.cdf(lower_v, b=b, scale=sigma)
    return 1 - prob_stable

# UI
st.title("⚡ Power Fluctuation Predictor")
st.markdown("Check voltage condition & get smart alerts")

# Voltage input
voltage = st.number_input("Enter Voltage (V):", min_value=0, max_value=300, step=1)

if st.button("Predict"):
    label_num = model.predict([[voltage]])[0]
    label = encoder.inverse_transform([label_num])[0]
    pv = calculate_pv(voltage, kappa, sigma)
    alert_needed = label == "Power Outage" or pv > 0.5

    st.markdown(f"### 📊 Voltage Input: `{voltage} V`")
    st.markdown(f"**Predicted Condition:** `{label}`")
    st.markdown(f"**Probability of Instability Pv(t):** `{round(pv * 100, 2)}%`")

    if alert_needed:
        st.error("⚠️ ALERT: Voltage is unstable or dangerous!")
    else:
        st.success("✅ Voltage is stable.")