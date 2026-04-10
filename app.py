# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Inflation Prediction API")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Automatically get feature names from scaler
FEATURE_NAMES = scaler.feature_names_in_.tolist()

print("Using Features:", FEATURE_NAMES)


# Input format
class InputData(BaseModel):
    features: list


# Home route
@app.get("/")
def home():
    return {"message": "Inflation Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: InputData):
    try:
        # Check number of inputs
        if len(data.features) != len(FEATURE_NAMES):
            return {
                "error": f"Expected {len(FEATURE_NAMES)} values but got {len(data.features)}"
            }

        # Convert to DataFrame with correct columns
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)

        # Ensure correct order (very important)
        input_df = input_df[FEATURE_NAMES]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        return {
            "predicted_inflation_rate": round(float(prediction[0]), 2)
        }

    except Exception as e:
        return {"error": str(e)}