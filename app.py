from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model, scaler = joblib.load("model/breast_cancer_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["radius"]),
            float(request.form["texture"]),
            float(request.form["perimeter"]),
            float(request.form["area"]),
            float(request.form["smoothness"]),
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        result = model.predict(features_scaled)[0]

        prediction = "Benign" if result == 1 else "Malignant"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

