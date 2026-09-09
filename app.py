

!pip install -q pandas numpy scikit-learn openpyxl joblib
     


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix)
     

FILE_PATH = "/content/hive_sensor_all_parameters_1min (1).csv"
print("Loading dataset...")
df = pd.read_csv(FILE_PATH)
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())
     

features = [
    "Hive_Temperature_C",
    "Hive_Relative_Humidity_%",
    "Hive_Weight_kg",
    "CO2_ppm",
    "Sound_Acoustic_Level",
    "Bee_Traffic_count_per_min",
    "Vibration_index",
    "Ambient_Temperature_C",
    "Ambient_Humidity_%",
    "Rainfall_mm"
]

target = "Health_Label"


# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

print("\nAll required columns are available.")

# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates().copy()

after = len(df)

print("\nDuplicates removed:", before - after)

# ============================================================
# 6. CONVERT SENSOR COLUMNS TO NUMERIC
# ============================================================

for col in features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================================
# 7. HANDLE INVALID NEGATIVE VALUES
# ============================================================

positive_columns = [
    "Hive_Weight_kg",
    "CO2_ppm",
    "Sound_Acoustic_Level",
    "Bee_Traffic_count_per_min",
    "Vibration_index",
    "Rainfall_mm"
]

for col in positive_columns:

    df.loc[df[col] < 0, col] = np.nan

# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before preprocessing:")

print(
    df[features].isnull().sum()
)

for col in features:

    median_value = df[col].median()

    df[col] = df[col].fillna(
        median_value
    )

print("\nMissing values after preprocessing:")

print(
    df[features].isnull().sum().sum()
)

# ============================================================
# 9. CLEAN TARGET LABEL
# ============================================================

df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.title()
)

print("\nHealth classes:")

print(
    df[target].value_counts()
)

     

# 10. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[target].copy()

print("\nX shape:", X.shape)

print("y shape:", y.shape)

# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

# ============================================================
# 12. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("\nFeature scaling completed.")

# ============================================================
# 13. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=12,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed!")

# ============================================================
# 14. MODEL PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)

# ============================================================
# 15. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# 16. SAVE DATA RANGES FOR UI
# ============================================================

data_ranges = {}

for col in features:

    data_ranges[col] = {

        "min": float(
            df[col].min()
        ),

        "max": float(
            df[col].max()
        ),

        "median": float(
            df[col].median()
        )
    }

# ============================================================
# 17. SAVE MODEL INFORMATION
# ============================================================

model_data = {

    "model": model,

    "scaler": scaler,

    "features": features,

    "target": target,

    "classes": list(
        model.classes_
    ),

    "data_ranges": data_ranges,

    "accuracy": accuracy
}

# ============================================================
# 18. SAVE MODEL
# ============================================================

MODEL_FILE = "hive_model.pkl"

joblib.dump(
    model_data,
    MODEL_FILE
)

print("\n====================================")
print("MODEL SAVED")
print("====================================")

print(
    f"Saved as: {MODEL_FILE}"
)

print(
    "File exists:",
    os.path.exists(MODEL_FILE)
)

print("\nClasses used by model:")

print(
    model.classes_
)

print("\nTraining completed successfully! 🐝")
     

# ============================================================
# TEST MODEL
# ============================================================

import joblib
import pandas as pd

model_data = joblib.load(
    "hive_model.pkl"
)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# Example sensor input
test_data = pd.DataFrame({
    "Hive_Temperature_C": [35],
    "Hive_Relative_Humidity_%": [60],
    "Hive_Weight_kg": [40],
    "CO2_ppm": [1000],
    "Sound_Acoustic_Level": [50],
    "Bee_Traffic_count_per_min": [100],
    "Vibration_index": [50],
    "Ambient_Temperature_C": [28],
    "Ambient_Humidity_%": [60],
    "Rainfall_mm": [0]
})

test_scaled = scaler.transform(
    test_data[features]
)
prediction = model.predict(
    test_scaled
)
probability = model.predict_proba(
    test_scaled
)
print("Prediction:", prediction[0])
print("\nProbabilities:")
for cls, prob in zip(
    model.classes_,
    probability[0]
):
    print(
        f"{cls}: {prob * 100:.2f}%"
    )
     

!pip install -q gradio
     

# ============================================================
# BEEHIVE HEALTH & HONEY YIELD
# SIMPLE UI DEMO
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "hive_model.pkl"

model_data = joblib.load(
    MODEL_FILE
)

model = model_data["model"]

scaler = model_data["scaler"]

features = model_data["features"]

data_ranges = model_data["data_ranges"]

# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(value, min_value, max_value):

    if max_value == min_value:
        return 50.0

    score = (
        (value - min_value)
        /
        (max_value - min_value)
    ) * 100

    return float(
        np.clip(score, 0, 100)
    )

# ============================================================
# HIVE STABILITY
# ============================================================

def hive_stability(
    temperature,
    humidity,
    weight
):

    temperature_score = max(
        0,
        100 - abs(
            temperature - 35
        ) * 10
    )

    humidity_score = max(
        0,
        100 - abs(
            humidity - 60
        ) * 2
    )

    weight_score = np.clip(
        (weight / 50) * 100,
        0,
        100
    )

    score = (

        temperature_score * 0.40 +

        humidity_score * 0.35 +

        weight_score * 0.25

    )

    return round(score)

# ============================================================
# BEE ACTIVITY
# ============================================================

def bee_activity(
    traffic,
    sound,
    vibration
):

    traffic_score = normalize(

        traffic,

        data_ranges[
            "Bee_Traffic_count_per_min"
        ]["min"],

        data_ranges[
            "Bee_Traffic_count_per_min"
        ]["max"]

    )

    sound_score = normalize(

        sound,

        data_ranges[
            "Sound_Acoustic_Level"
        ]["min"],

        data_ranges[
            "Sound_Acoustic_Level"
        ]["max"]

    )

    vibration_score = normalize(

        vibration,

        data_ranges[
            "Vibration_index"
        ]["min"],

        data_ranges[
            "Vibration_index"
        ]["max"]

    )

    score = (

        traffic_score * 0.50 +

        sound_score * 0.30 +

        vibration_score * 0.20

    )

    return round(score)

# ============================================================
# ENVIRONMENTAL HEALTH
# ============================================================

def environmental_health(
    temperature,
    humidity,
    co2
):

    temperature_score = max(
        0,
        100 - abs(
            temperature - 35
        ) * 10
    )

    humidity_score = max(
        0,
        100 - abs(
            humidity - 60
        ) * 2
    )

    co2_score = max(
        0,
        100 - abs(
            co2 - 1000
        ) / 20
    )

    score = (

        temperature_score * 0.40 +

        humidity_score * 0.30 +

        co2_score * 0.30

    )

    return round(
        np.clip(score, 0, 100)
    )

# ============================================================
# WEATHER STRESS
# ============================================================

def weather_stress(
    ambient_temperature,
    ambient_humidity,
    rainfall
):

    temperature_stress = min(
        100,
        abs(
            ambient_temperature - 28
        ) * 8
    )

    humidity_stress = min(
        100,
        abs(
            ambient_humidity - 60
        ) * 1.5
    )

    rainfall_stress = min(
        100,
        rainfall * 10
    )

    score = (

        temperature_stress * 0.40 +

        humidity_stress * 0.30 +

        rainfall_stress * 0.30

    )

    return round(
        np.clip(score, 0, 100)
    )

# ============================================================
# ANOMALY
# ============================================================

def anomaly_result(
    health_score,
    stability,
    activity,
    environment
):

    risk = (

        (100 - health_score) * 0.35 +

        (100 - stability) * 0.25 +

        (100 - activity) * 0.20 +

        (100 - environment) * 0.20

    )

    risk = round(
        np.clip(risk, 0, 100)
    )

    if risk < 30:

        level = "LOW"

    elif risk < 60:

        level = "MEDIUM"

    else:

        level = "HIGH"

    return risk, level

# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_hive(

    hive_temperature,

    hive_humidity,

    hive_weight,

    co2,

    sound,

    bee_traffic,

    vibration,

    ambient_temperature,

    ambient_humidity,

    rainfall

):

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Hive_Temperature_C":
            [hive_temperature],

        "Hive_Relative_Humidity_%":
            [hive_humidity],

        "Hive_Weight_kg":
            [hive_weight],

        "CO2_ppm":
            [co2],

        "Sound_Acoustic_Level":
            [sound],

        "Bee_Traffic_count_per_min":
            [bee_traffic],

        "Vibration_index":
            [vibration],

        "Ambient_Temperature_C":
            [ambient_temperature],

        "Ambient_Humidity_%":
            [ambient_humidity],

        "Rainfall_mm":
            [rainfall]
    })

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        input_data[features]
    )

    probabilities = model.predict_proba(
        scaled_data
    )[0]

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    healthy_probability = probability_dict.get(
        "Healthy",
        0
    )

    warning_probability = probability_dict.get(
        "Warning",
        0
    )

    critical_probability = probability_dict.get(
        "Critical",
        0
    )

    # --------------------------------------------------------
    # HEALTH SCORE
    # --------------------------------------------------------

    health_score = (

        healthy_probability * 100 +

        warning_probability * 60 +

        critical_probability * 10

    )

    health_score = round(
        np.clip(
            health_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # HEALTH STATUS
    # --------------------------------------------------------

    if health_score >= 70:

        status = "🟢 HEALTHY"

    elif health_score >= 40:

        status = "🟡 WARNING"

    else:

        status = "🔴 CRITICAL"

    # --------------------------------------------------------
    # OTHER SCORES
    # --------------------------------------------------------

    stability = hive_stability(

        hive_temperature,

        hive_humidity,

        hive_weight

    )

    activity = bee_activity(

        bee_traffic,

        sound,

        vibration

    )

    environment = environmental_health(

        hive_temperature,

        hive_humidity,

        co2

    )

    weather = weather_stress(

        ambient_temperature,

        ambient_humidity,

        rainfall

    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    anomaly_score, anomaly_level = anomaly_result(

        health_score,

        stability,

        activity,

        environment

    )

    # --------------------------------------------------------
    # YIELD SCORE
    # --------------------------------------------------------

    yield_score = (

        health_score * 0.35 +

        stability * 0.20 +

        activity * 0.25 +

        environment * 0.15 +

        (100 - weather) * 0.05

    )

    yield_score = round(
        np.clip(
            yield_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # EXPECTED YIELD
    # --------------------------------------------------------

    if yield_score >= 70:

        expected_yield = "🍯 HIGH"

    elif yield_score >= 40:

        expected_yield = "🟡 MEDIUM"

    else:

        expected_yield = "🔴 LOW"

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = f"""
# 🐝 BEEHIVE HEALTH REPORT

## 🏥 Health Status

**Health Score:** {health_score} / 100

**Status:** {status}

---

## 🐝 Hive Performance

**Hive Stability:** {stability} / 100

**Bee Activity:** {activity} / 100

**Environmental Health:** {environment} / 100

**Weather Stress:** {weather} / 100

---

## 🚨 Anomaly Monitoring

**Anomaly Score:** {anomaly_score} / 100

**Anomaly Level:** {anomaly_level}

---

## 🍯 Honey Yield Prediction

**Expected Yield:** {expected_yield}

**Yield Score:** {yield_score} / 100

---

### Model Prediction Probabilities

Healthy: {healthy_probability * 100:.2f}%

Warning: {warning_probability * 100:.2f}%

Critical: {critical_probability * 100:.2f}%
"""

    return output

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Beehive Health Prediction"
) as demo:

    gr.Markdown(
        """
        # 🐝 Beehive Health & Honey Yield Prediction

        Enter the current sensor values to predict
        **Hive Health, Health Score, Anomaly Level and
        Expected Honey Yield.**
        """
    )

    # --------------------------------------------------------
    # HIVE INPUTS
    # --------------------------------------------------------

    gr.Markdown(
        "## 🏠 Hive Conditions"
    )

    with gr.Row():

        hive_temperature = gr.Number(
            label="Hive Temperature (°C)",
            value=35
        )

        hive_humidity = gr.Number(
            label="Hive Humidity (%)",
            value=60
        )

        hive_weight = gr.Number(
            label="Hive Weight (kg)",
            value=40
        )

    with gr.Row():

        co2 = gr.Number(
            label="CO₂ (ppm)",
            value=1000
        )

        sound = gr.Number(
            label="Sound / Acoustic Level",
            value=50
        )

    # --------------------------------------------------------
    # BEE ACTIVITY
    # --------------------------------------------------------

    gr.Markdown(
        "## 🐝 Bee Activity"
    )

    with gr.Row():

        bee_traffic = gr.Number(
            label="Bee Traffic / min",
            value=100
        )

        vibration = gr.Number(
            label="Vibration Index",
            value=50
        )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    gr.Markdown(
        "## 🌦️ Weather Conditions"
    )

    with gr.Row():

        ambient_temperature = gr.Number(
            label="Ambient Temperature (°C)",
            value=28
        )

        ambient_humidity = gr.Number(
            label="Ambient Humidity (%)",
            value=60
        )

        rainfall = gr.Number(
            label="Rainfall (mm)",
            value=0
        )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    predict_button = gr.Button(
        "🔍 PREDICT HIVE HEALTH",
        variant="primary"
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = gr.Markdown(
        """
        ### Result will appear here
        """
    )

    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    predict_button.click(

        fn=predict_hive,

        inputs=[

            hive_temperature,

            hive_humidity,

            hive_weight,

            co2,

            sound,

            bee_traffic,

            vibration,

            ambient_temperature,

            ambient_humidity,

            rainfall

        ],

        outputs=output

    )

# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=True
)
     
Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
* Running on public URL: https://a5dc51d3dda91ff707.gradio.live

This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
Keyboard interruption in main thread... closing server.
Killing tunnel 127.0.0.1:7860 <> https://a5dc51d3dda91ff707.gradio.live

from google.colab import drive
import os
import pickle

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Create the directory if it doesn't exist
output_dir = '/content/drive/My Drive/hive_project'
os.makedirs(output_dir, exist_ok=True)

# 3. Save the files
df.to_pickle(os.path.join(output_dir, 'hive_dataframe.pkl'))

with open(os.path.join(output_dir, 'hive_model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print("Files successfully saved to Google Drive!")
     
Mounted at /content/drive
Files successfully saved to Google Drive!

import pandas as pd

# 1. Save the dataframe to a pickle file first (using df from your notebook)
df.to_pickle('hive_dataframe.pkl')

# 2. Now try loading it back to verify
loaded_df = pd.read_pickle('hive_dataframe.pkl')
print("Pickle file saved and loaded successfully! Shape:", loaded_df.shape)
     
Pickle file saved and loaded successfully! Shape: (20160, 13)

loaded_df = pd.read_pickle('/content/drive/My Drive/hive_project/hive_dataframe.pkl')
print("Loaded from Drive successfully! Shape:", loaded_df.shape)
     
Loaded from Drive successfully! Shape: (20160, 13)

import os
import pickle
import joblib

# 1. Ensure the directory exists in Google Drive
output_dir = '/content/drive/My Drive/hive_project'
os.makedirs(output_dir, exist_ok=True)

# 2. Save the DataFrame as a pickle file
loaded_df.to_pickle(os.path.join(output_dir, 'hive_dataframe.pkl'))

# 3. Save the model data dictionary using joblib/pickle
joblib.dump(model_data, os.path.join(output_dir, 'hive_model.pkl'))

print("Everything is successfully saved to your Google Drive folder!")
     
Everything is successfully saved to your Google Drive folder!

import os

# Check if the pickle file exists in your current Colab session folder
print("Exists in Colab local folder?", os.path.exists('hive_dataframe.pkl'))

# Check if it exists in your Google Drive folder
print("Exists in Google Drive?", os.path.exists('/content/drive/My Drive/hive_project/hive_dataframe.pkl'))

     
Exists in Colab local folder? True
Exists in Google Drive? True
