import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to estimate churn risk.")

# Load model package
with open("churn_model.pkl", "rb") as f:
    pkg = pickle.load(f)

model = pkg['model']
scaler = pkg['scaler']
model_columns = pkg['columns']
categorical_cols = pkg['categorical_cols']

def predict_new_customer_streamlit(input_dict):
    df_temp = pd.DataFrame([input_dict])
    df_temp_encoded = pd.get_dummies(df_temp, columns=categorical_cols, drop_first=True)

    for col in model_columns:
        if col not in df_temp_encoded.columns:
            df_temp_encoded[col] = 0

    df_temp_encoded = df_temp_encoded[model_columns]
    scaled = scaler.transform(df_temp_encoded)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    return pred, prob

# --- Input form ---

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=300.0)

if st.button("Predict Churn"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'InternetService': internet,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    pred, prob = predict_new_customer_streamlit(input_data)

    if pred == 1:
        st.error(f"⚠ High risk of churn. Probability: {prob*100:.2f}%")
    else:
        st.success(f"Customer likely to stay. Churn risk: {prob*100:.2f}%")
