import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('depression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess user input
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
    features = scaler.transform(features)
    return features

# ---- LOGIN SYSTEM ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.set_page_config(page_title="Depression Prediction", page_icon="🧠", layout="centered")
    st.markdown("<h1 style='text-align:center;'>🔑 Student Login</h1>", unsafe_allow_html=True)
    
    name = st.text_input("👤 Name")
    gender = st.selectbox("⚥ Gender", ["Male", "Female"])
    
    if st.button("Login", use_container_width=True):
        if name.strip():
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.session_state.user_gender = gender
        else:
            st.warning("⚠️ Please enter your name.")
    st.stop()

# ---- MAIN APP ----
st.set_page_config(page_title="Depression Prediction", page_icon="🧠", layout="centered")

# Header with Logout Button
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"<h2>🧠 Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

st.markdown("### 📋 Fill in the details below:")

# Input fields in two columns for better layout
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🎂 Age", min_value=10, max_value=30, value=20)
    academic_pressure = st.slider("📚 Academic Pressure (0-9)", 0, 9, 5)
    study_satisfaction = st.slider("😊 Study Satisfaction (0-9)", 0, 9, 5)
    sleep_duration = st.slider("🛏 Sleep Duration (hours)", 0, 12, 6)
    dietary = st.selectbox("🥗 Dietary Habits", ["Good", "Poor"])
with col2:
    suicidal = st.selectbox("💭 Suicidal Thoughts?", ["No", "Yes"])
    study_hours = st.slider("⏳ Study Hours per day", 0, 12, 4)
    fin_stress = st.selectbox("💰 Financial Stress", ["Low", "High"])
    family_history = st.selectbox("👨‍👩‍👧 Family History of Mental Illness", ["No", "Yes"])

st.markdown("---")

# Prediction Button
if st.button("🔍 Predict", use_container_width=True):
    input_data = {
        "Gender": st.session_state.user_gender,
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
