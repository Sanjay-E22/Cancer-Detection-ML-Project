import streamlit as st
import numpy as np
import pickle
import os

st.write("contents of root:",os.listdir("."))
st.write("contents of model folder:",os.listdir("model"))

# Load model
with open("model/cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔬 Breast Cancer Prediction App")

st.write("Enter cell features below:")

# User Inputs (first 10 features)
features = []
feature_names = [
    "mean radius", "mean texture", "mean perimeter",
    "mean area", "mean smoothness", "mean compactness",
    "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension"
]

for name in feature_names:
    val = st.slider(name, 0.0, 100.0, 1.0)
    features.append(val)

# Predict
if st.button("Predict"):
    features_np = np.array([features])
    pred = model.predict(features_np)[0]
    if pred == 1:
        st.success("✅ Diagnosis: Benign (Non-cancerous)")
    else:
        st.error("⚠ Diagnosis: Malignant (Cancerous)")