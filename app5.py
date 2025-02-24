import streamlit as st
import pickle
import numpy as np
import os

# Load Model and Scaler with Path Handling
model_path = os.path.join("model5", "trained_model5.pkl")
scaler_path = os.path.join("model5", "scaler5.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found! Expected at: {model_path}")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file not found! Expected at: {scaler_path}")
    st.stop()

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title
st.title("💤 Sleep Disorder Prediction App")
st.write("🌙 Enter your details to check for sleep disorder risk.")

# User Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
gender = gender_mapping[gender]

# Occupation
occupation = st.selectbox("Occupation", ["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Accountant", "Salesperson", "Others"])
occupation_mapping = {"Nurse": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3, "Teacher": 4, "Accountant": 5, "Salesperson": 6, "Others": 7}
occupation = occupation_mapping[occupation]

sleep_duration = st.slider("Sleep Duration (hours)", 1.0, 12.0, 7.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
physical_activity = st.number_input("Physical Activity Level (minutes/day)", 0, 300, value=30)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

# BMI Category
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
bmi_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
bmi_category = bmi_mapping[bmi_category]

heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, value=70)
daily_steps = st.number_input("Daily Steps", 0, 20000, value=5000)
systolic = st.number_input("Systolic Blood Pressure", 80, 200, value=120)
diastolic = st.number_input("Diastolic Blood Pressure", 50, 130, value=80)

# Prepare Data
input_features = np.array([[age, gender, occupation, sleep_duration, quality_of_sleep, 
                            physical_activity, stress_level, bmi_category, heart_rate, 
                            daily_steps, systolic, diastolic]])

# Scale the input
input_features = scaler.transform(input_features)

# Sleep Disorders Information
disorder_info = {
    "Insomnia": ("Difficulty falling or staying asleep.", "Reduce caffeine, maintain a sleep schedule, try relaxation techniques."),
    "Sleep Anxiety": ("Anxiety-related sleep disturbances.", "Practice meditation, avoid screens before bed, deep breathing exercises."),
    "Obstructive Sleep Apnea": ("Breathing stops during sleep due to airway blockage.", "Lose weight, avoid alcohol, consider CPAP therapy."),
    "Hypertension-related Sleep Issues": ("Poor sleep linked to high blood pressure.", "Monitor BP, reduce salt, maintain a balanced diet."),
    "Restless Leg Syndrome": ("Uncontrollable urge to move legs, worse at night.", "Exercise, avoid caffeine, maintain a regular sleep schedule."),
    "Narcolepsy": ("Excessive daytime sleepiness, sudden sleep attacks.", "Maintain a consistent schedule, avoid heavy meals before bed."),
}

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_features)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of sleep disorder detected! Consult a doctor.")

        # Suggest possible disorders based on inputs
        possible_disorders = []
        
        if sleep_duration < 5 or quality_of_sleep < 3:
            possible_disorders.append("Insomnia")
        
        if stress_level > 7:
            possible_disorders.append("Sleep Anxiety")
        
        if bmi_category == 3 and heart_rate > 90:
            possible_disorders.append("Obstructive Sleep Apnea")
        
        if systolic > 140 or diastolic > 90:
            possible_disorders.append("Hypertension-related Sleep Issues")
        
        if daily_steps < 3000 and physical_activity < 20:
            possible_disorders.append("Restless Leg Syndrome")
        
        if sleep_duration > 9 and stress_level < 3:
            possible_disorders.append("Narcolepsy")

        if not possible_disorders:
            possible_disorders.append("General Sleep Disorder")

        st.warning(f"🛏️ **Possible Sleep Disorders:** {', '.join(possible_disorders)}")

        for disorder in possible_disorders:
            if disorder in disorder_info:
                st.subheader(f"🩺 {disorder}")
                st.write(f"🔹 **Definition:** {disorder_info[disorder][0]}")
                st.write(f"💡 **Tips:** {disorder_info[disorder][1]}")

    else:
        st.success("✅ Low risk of sleep disorder. Keep maintaining good habits!")

st.write("📌 **Note:** This AI prediction should not replace professional medical advice.")
