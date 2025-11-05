# --- main.py ---
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from prophet import Prophet
import os

# Create the app instance
app = FastAPI(
    title="Demand Forecast API",
    description="Returns future demand forecasts from a trained Prophet model",
    version="1.0"
)

# --- Load the trained model ---
MODEL_PATH = "../models/prophet_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# --- Root endpoint ---
@app.get("/")
def home():
    return {"message": "Welcome to the Demand Forecast API! Use /forecast?weeks=12 to get predictions."}

# --- Forecast endpoint ---
@app.get("/forecast")
def forecast(weeks: int = 12):
    """
    Generate demand forecasts for the next `weeks` weeks.
    """
    try:
        # Generate future dataframe
        future = model.make_future_dataframe(periods=weeks, freq="W")
        forecast = model.predict(future).tail(weeks)

        # Return only the new predictions
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
        return {"forecast": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
