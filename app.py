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
feature_names = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'fractal_dimension_mean',
 'radius_se',
 'texture_se',
 'perimeter_se',
 'area_se',
 'smoothness_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'symmetry_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']

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