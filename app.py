import os
from fastapi import FastAPI, Header, HTTPException, Depends
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
# API KEY AUTH
# ============================================================

API_KEY = os.environ.get("API_KEY")

def verify_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# TEMPORARY DEBUG ENDPOINT - remove after fixing the key mismatch
@app.get("/debug-key")
def debug_key():
    if not API_KEY:
        return {"configured": False}
    return {
        "configured": True,
        "length": len(API_KEY),
        "first_4": API_KEY[:4],
        "last_4": API_KEY[-4:],
    }

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

# ============================================================
# SCORING FUNCTIONS
# ============================================================

def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 50.0
    score = ((value - min_value) / (max_value - min_value)) * 100
    return float(np.clip(score, 0, 100))

def hive_stability(temperature, humidity, weight):
    temperature_score = max(0, 100 - abs(temperature - 35) * 10)
    humidity_score = max(0, 100 - abs(humidity - 60) * 2)
    weight_score = np.clip((weight / 50) * 100, 0, 100)
    score = temperature_score * 0.40 + humidity_score * 0.35 + weight_score * 0.25
    return round(score)

def bee_activity(traffic, sound, vibration):
    traffic_score = normalize(traffic, data_ranges["Bee_Traffic_count_per_min"]["min"], data_ranges["Bee_Traffic_count_per_min"]["max"])
    sound_score = normalize(sound, data_ranges["Sound_Acoustic_Level"]["min"], data_ranges["Sound_Acoustic_Level"]["max"])
    vibration_score = normalize(vibration, data_ranges["Vibration_index"]["min"], data_ranges["Vibration_index"]["max"])
    score = traffic_score * 0.50 + sound_score * 0.30 + vibration_score * 0.20
    return round(score)

def environmental_health(temperature, humidity, co2):
    temperature_score = max(0, 100 - abs(temperature - 35) * 10)
    humidity_score = max(0, 100 - abs(humidity - 60) * 2)
    co2_score = max(0, 100 - abs(co2 - 1000) / 20)
    score = temperature_score * 0.40 + humidity_score * 0.30 + co2_score * 0.30
    return round(np.clip(score, 0, 100))

def weather_stress(ambient_temperature, ambient_humidity, rainfall):
    temperature_stress = min(100, abs(ambient_temperature - 28) * 8)
    humidity_stress = min(100, abs(ambient_humidity - 60) * 1.5)
    rainfall_stress = min(100, rainfall * 10)
    score = temperature_stress * 0.40 + humidity_stress * 0.30 + rainfall_stress * 0.30
    return round(np.clip(score, 0, 100))

def anomaly_result(health_score, stability, activity, environment):
    risk = ((100 - health_score) * 0.35 + (100 - stability) * 0.25 +
            (100 - activity) * 0.20 + (100 - environment) * 0.20)
    risk = round(np.clip(risk, 0, 100))
    if risk < 30:
        level = "LOW"
    elif risk < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return risk, level

# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_hive(data: HiveInput):
    input_data = pd.DataFrame({
        "Hive_Temperature_C": [data.hive_temperature],
        "Hive_Relative_Humidity_%": [data.hive_humidity],
        "Hive_Weight_kg": [data.hive_weight],
        "CO2_ppm": [data.co2],
        "Sound_Acoustic_Level": [data.sound],
        "Bee_Traffic_count_per_min": [data.bee_traffic],
        "Vibration_index": [data.vibration],
        "Ambient_Temperature_C": [data.ambient_temperature],
        "Ambient_Humidity_%": [data.ambient_humidity],
        "Rainfall_mm": [data.rainfall],
    })

    scaled_data = scaler.transform(input_data[features])
    probabilities = model.predict_proba(scaled_data)[0]
    probability_dict = dict(zip(model.classes_, probabilities))

    healthy_p = probability_dict.get("Healthy", 0)
    warning_p = probability_dict.get("Warning", 0)
    critical_p = probability_dict.get("Critical", 0)

    health_score = round(np.clip(healthy_p * 100 + warning_p * 60 + critical_p * 10, 0, 100))

    if health_score >= 70:
        status = "HEALTHY"
    elif health_score >= 40:
        status = "WARNING"
    else:
        status = "CRITICAL"

    stability = hive_stability(data.hive_temperature, data.hive_humidity, data.hive_weight)
    activity = bee_activity(data.bee_traffic, data.sound, data.vibration)
    environment = environmental_health(data.hive_temperature, data.hive_humidity, data.co2)
    weather = weather_stress(data.ambient_temperature, data.ambient_humidity, data.rainfall)

    anomaly_score, anomaly_level = anomaly_result(health_score, stability, activity, environment)

    yield_score = round(np.clip(
        health_score * 0.35 + stability * 0.20 + activity * 0.25 +
        environment * 0.15 + (100 - weather) * 0.05,
        0, 100
    ))

    if yield_score >= 70:
        expected_yield = "HIGH"
    elif yield_score >= 40:
        expected_yield = "MEDIUM"
    else:
        expected_yield = "LOW"

    return {
        "health_score": health_score,
        "status": status,
        "hive_stability": stability,
        "bee_activity": activity,
        "environmental_health": environment,
        "weather_stress": weather,
        "anomaly_score": anomaly_score,
        "anomaly_level": anomaly_level,
        "expected_yield": expected_yield,
        "yield_score": yield_score,
        "probabilities": {
            "Healthy": round(healthy_p * 100, 2),
            "Warning": round(warning_p * 100, 2),
            "Critical": round(critical_p * 100, 2),
        }
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>🐝 Beehive Health Prediction API is running. POST to /predict</h2>"
