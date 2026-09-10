import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "hive_model.pkl"
model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
data_ranges = model_data["data_ranges"]

app = FastAPI(title="Beehive Health Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# API KEY AUTH  <-- NEW
# ============================================================

API_KEY = os.environ.get("API_KEY")

def verify_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        # Fails safe: if you forgot to set the env var, block everything
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ============================================================
# INPUT SCHEMA
# ============================================================

class HiveInput(BaseModel):
    hive_temperature: float
    hive_humidity: float
    hive_weight: float
    co2: float
    sound: float
    bee_traffic: float
    vibration: float
    ambient_temperature: float
    ambient_humidity: float
    rainfall: float

# ... (all your scoring functions stay exactly the same) ...

# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

from fastapi import Depends  # <-- NEW import

@app.post("/predict", dependencies=[Depends(verify_api_key)])  # <-- CHANGED
def predict_hive(data: HiveInput):
    ...  # unchanged body

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>🐝 Beehive Health Prediction API is running. POST to /predict</h2>"
