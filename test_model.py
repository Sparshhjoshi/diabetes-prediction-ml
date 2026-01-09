import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# Create input with correct feature names
sample = pd.DataFrame([{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 120,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}])

# Predict
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Person is Diabetic")
else:
    print("Person is NOT Diabetic")