import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('depression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define function to preprocess user input
def preprocess_input(data):
    gender = 0 if data["Gender"] == "Male" else 1
    dietary = 1 if data["Dietary Habits"] == "Good" else 0
    suicide = 1 if data["Suicidal Thoughts"] == "Yes" else 0
    fam_history = 1 if data["Family History"] == "Yes" else 0
    fin_stress = 1 if data["Financial Stress"] == "High" else 0

    features = np.array([[
        gender,
        data["Age"],
        data["Academic Pressure"],
        data["Study Satisfaction"],
        data["Sleep Duration"],
        dietary,
        suicide,
        data["Study Hours"],
        fin_stress,
        fam_history
    ]])

    # Scale the features
    features = scaler.transform(features)
    return features

# Streamlit UI
st.title("🧠 Student Depression Prediction App")

st.write("Fill in the details below:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=30, value=20)
academic_pressure = st.slider("Academic Pressure (0-9)", 0, 9, 5)
study_satisfaction = st.slider("Study Satisfaction (0-9)", 0, 9, 5)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 6)
dietary = st.selectbox("Dietary Habits", ["Good", "Poor"])
suicidal = st.selectbox("Have you had suicidal thoughts?", ["No", "Yes"])
study_hours = st.slider("Study Hours per day", 0, 12, 4)
fin_stress = st.selectbox("Financial Stress", ["Low", "High"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

# Submit button
if st.button("Predict"):
    input_data = {
        "Gender": gender,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary,
        "Suicidal Thoughts": suicidal,
        "Study Hours": study_hours,
        "Financial Stress": fin_stress,
        "Family History": family_history
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)

    if prediction[0] == 1:
        st.error("⚠️ The student is likely to be **Depressed**.")
    else:
        st.success("✅ The student is **Not Likely Depressed**.")
