from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get form data
        data = {
            "Pregnancies": int(request.form["pregnancies"]),
            "Glucose": float(request.form["glucose"]),
            "BloodPressure": float(request.form["bloodpressure"]),
            "SkinThickness": float(request.form["skinthickness"]),
            "Insulin": float(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DiabetesPedigreeFunction": float(request.form["dpf"]),
            "Age": int(request.form["age"])
        }

        # Convert to DataFrame (IMPORTANT)
        input_df = pd.DataFrame([data])

        # Predict
        result = model.predict(input_df)[0]
        prediction = "Diabetic" if result == 1 else "Not Diabetic"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)