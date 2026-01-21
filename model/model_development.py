# model/model_development.py

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target  # 0 = malignant, 1 = benign

# 2. Feature selection (ONLY allowed features)
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

X = df[selected_features]
y = df["diagnosis"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature scaling (MANDATORY for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# 7. Save model and scaler
joblib.dump((model, scaler), "model/breast_cancer_model.pkl")

# 8. Reload model (NO retraining)
loaded_model, loaded_scaler = joblib.load("model/breast_cancer_model.pkl")

# Test reload
sample = X_test.iloc[0].values.reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print("Reloaded model prediction:", prediction)
