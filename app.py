
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("AI Student Performance Prediction System")
st.write("Enter student details to predict performance.")

attendance = st.number_input(
    "Attendance (%)", min_value=0, max_value=100, value=85
)

internal_marks = st.number_input(
    "Internal Marks", min_value=0, max_value=100, value=75
)

assignment_score = st.number_input(
    "Assignment Score", min_value=0, max_value=100, value=80
)

study_hours = st.number_input(
    "Study Hours per Day", min_value=0, max_value=24, value=4
)

if st.button("Predict Performance"):
    student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours
    }])

    prediction = model.predict(student)[0]

    st.success(f"Predicted Performance: {prediction}")
