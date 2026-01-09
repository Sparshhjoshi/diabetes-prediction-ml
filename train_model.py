# ===============================
# Diabetes Detection - Model Training
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset/diabetes.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# -------------------------------
# 2. Data Cleaning
# -------------------------------
# These columns cannot have 0 values in real life
columns_with_zero = ["Glucose", "BloodPressure", "BMI", "Insulin"]

for col in columns_with_zero:
    df[col] = df[col].replace(0, df[col].mean())

# -------------------------------
# 3. Split Features and Target
# -------------------------------
X = df.drop("Outcome", axis=1)   # Features
y = df["Outcome"]                # Label

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 5. Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
joblib.dump(model, "model.pkl")
print("\nModel saved successfully as model.pkl")