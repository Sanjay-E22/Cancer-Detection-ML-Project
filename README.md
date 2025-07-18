# Cancer-Detection-ML-Project
# 🧬 Cancer Detection using Logistic Regression

This project implements a *Logistic Regression* model to detect cancer based on patient data. The model is trained using the liblinear solver and aims to predict the *diagnosis* column in the dataset (typically malignant or benign tumors).

---

## 📌 Project Overview

- *Objective*: Predict whether a tumor is benign or malignant using logistic regression.
- *Target column*: diagnosis
- *Model used*: LogisticRegression(solver='liblinear')
- *Preprocessing*: MinMaxScaler, Train-Test Split
- *App framework: Streamlit
- *Depoyment: Streamlit Cloud

---

## 🧪 Dataset

The dataset used is typically the *Breast Cancer Wisconsin (Diagnostic) Dataset*. You can load it via sklearn.datasets.load_breast_cancer() or use a CSV file.

- Features: Various measurements of cell nuclei (radius, texture, perimeter, area, etc.)
- Target: diagnosis — M = Malignant, B = Benign

---
## 🌐 Live Demo
 https://cancer-detection-ml-project-gbt6cffzmp6h5sswd6ctfg.streamlit.app/
## 🔧 Libraries Used

```python
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
