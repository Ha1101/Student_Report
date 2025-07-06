import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load trained models
math_model = joblib.load("catboost_math_model.pkl")
port_model = joblib.load("catboost_portuguese_model.pkl")

# Title
st.title("🎓 Student Performance Predictor")
st.markdown("Predict the final grade (G3) for a student based on academic and personal features.")

# Subject selection
subject = st.radio("Choose Subject Dataset:", ["Math", "Portuguese"])

# Input form
st.subheader("📋 Student Features")

studytime = st.slider("Study Time (1 = <2 hrs, 4 = >10 hrs)", 1, 4, 2)
Medu = st.selectbox("Mother's Education Level", [0, 1, 2, 3, 4])
Fedu = st.selectbox("Father's Education Level", [0, 1, 2, 3, 4])
internet = st.radio("Has Internet Access?", ["Yes", "No"])
higher = st.radio("Aspires for Higher Education?", ["Yes", "No"])
paid = st.radio("Attends Paid Extra Classes?", ["Yes", "No"])
Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
G1 = st.slider("Grade in Period 1 (G1)", 0, 20, 10)
G2 = st.slider("Grade in Period 2 (G2)", 0, 20, 10)

# Categorical encoding
internet_val = 1 if internet == "Yes" else 0
higher_val = 1 if higher == "Yes" else 0
paid_val = 1 if paid == "Yes" else 0
Mjob_map = {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4}
Mjob_val = Mjob_map[Mjob]

# Create feature DataFrame
features = pd.DataFrame([{
    'G1': G1,
    'G2': G2,
    'studytime': studytime,
    'Medu': Medu,
    'Fedu': Fedu,
    'internet_availability': internet_val,
    'higher_education': higher_val,
    'paid_classes': paid_val,
    'Mjob': Mjob_val
}])

# Predict and show result
if st.button("Predict Final Grade (G3)"):
    model = math_model if subject == "Math" else port_model
    expected_features = model.feature_names_
    features = features.reindex(columns=expected_features)

    try:
        prediction = model.predict(features)[0]
        st.success(f"🎯 Predicted Final Grade (G3): **{prediction:.2f}** / 20")
        st.write("📑 Input Summary:")
        st.dataframe(features)

        # Visualization: G1 vs G2 vs G3 Scatterplot
        st.subheader("📊 G1 vs G2 vs Final Grade (G3) Scatter Plot")

        # Load dataset for plotting
        if subject == "Math":
            df = pd.read_csv("student/student-mat.csv", sep=";")
        else:
            df = pd.read_csv("student/student-por.csv", sep=";")

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot existing data
        ax.scatter(df['G1'], df['G2'], df['G3'], c='blue', label='Existing Students')

        # Plot prediction point
        ax.scatter(G1, G2, prediction, c='red', s=150, label='Your Prediction')

        ax.set_xlabel("G1 (Period 1 Grade)")
        ax.set_ylabel("G2 (Period 2 Grade)")
        ax.set_zlabel("G3 (Final Grade)")
        ax.set_title(f"Student Performance - {subject} Dataset")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Expected features:", expected_features)
