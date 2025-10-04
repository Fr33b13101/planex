import json
import numpy as np
import pandas as pd
import tensorflow as tf
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.units import Quantity 


# -------------------------------
# Custom serializer for JSON
# -------------------------------
from astropy.units import Quantity
from astropy.table import MaskedColumn

def custom_serializer(obj):
    """Converts non-serializable objects to JSON-safe formats."""
    if isinstance(obj, (np.str_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Quantity):
        return f"{obj.value} {obj.unit}"
    # Handle masked values from astropy tables
    if hasattr(obj, 'mask'):
        if obj.mask:
            return None
        return float(obj)
    if obj is None or obj == "--":
        return None
    raise TypeError(f"Type {type(obj)} not serializable")   


# -------------------------------
# Kepler API utility
# -------------------------------
def get_kepid_classification(kepid):
    """Fetch classification data for a given Kepler ID and return as JSON."""
    table = NasaExoplanetArchive.query_criteria(
        table="q1_q17_dr25_koi",
        where=f"kepid={kepid}"
    )

    if len(table) == 0:
        return {"error": f"No record found for KepID {kepid}."}

    result = {
        "kepid": kepid,
        "koi_name": str(table["kepoi_name"][0]),
        "disposition": str(table["koi_disposition"][0]),
        "period_days": custom_serializer(table["koi_period"][0]),
        "radius_earth": custom_serializer(table["koi_prad"][0]),
        "stellar_temp": custom_serializer(table["koi_steff"][0])
    }


    # Return the JSON object directly instead of saving to file
    return result

# -------------------------------
# Kepler API endpoint
# -------------------------------


# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "exominer_like_keras.keras"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="ExoMiner Classification API")

# Allow cross-origin requests (helps prevent 405 on browser preflight OPTIONS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# JSON input schema
# -------------------------------
class ExoMinerInput(BaseModel):
    lc_local: list[float]
    lc_global: list[float]
    lc_unfolded: list[float]
    centroid: list[float]
    scalar_features: list[float]

# -------------------------------
# JSON-based prediction endpoint
# -------------------------------
@app.post("/predict/json")
async def predict_from_json(data: ExoMinerInput):
    try:
        # Convert lists from frontend JSON into correctly-shaped NumPy arrays
        lc_local = np.full((1, 201, 1), data.lc_local, dtype=float)
        lc_global = np.full((1, 2001, 1), data.lc_global, dtype=float)
        lc_unfolded = np.full((1, 4000, 1), data.lc_unfolded, dtype=float)
        centroid = np.full((1, 2001, 1), data.centroid, dtype=float)
        scalar_features = np.full((1, 30), data.scalar_features, dtype=float)

        # Make prediction
        preds = model.predict([lc_local, lc_global, lc_unfolded, centroid, scalar_features])
        prob = float(preds[0][1])  # probability of Planet Candidate
        label = "Planet Candidate" if prob >= 0.5 else "False Positive"

        # Convert probability to percentage
        confidence_percent = round(prob * 100, 2)

        return {"class": label, "confidence": confidence_percent}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

        return JSONResponse(status_code=400, content={"error": str(e)})

# -------------------------------
# CSV prediction endpoint
# -------------------------------
@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_cols = ["lc_local", "lc_global", "lc_unfolded", "centroid", "scalar_features"]
        for col in required_cols:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Missing column: {col}"})

        num_records = len(df)

        lc_local = np.array([np.full((201, 1), val, dtype=float) for val in df["lc_local"]])
        lc_global = np.array([np.full((2001, 1), val, dtype=float) for val in df["lc_global"]])
        lc_unfolded = np.array([np.full((4000, 1), val, dtype=float) for val in df["lc_unfolded"]])
        centroid = np.array([np.full((2001, 1), val, dtype=float) for val in df["centroid"]])
        scalar_features = np.array([np.full(30, val, dtype=float) for val in df["scalar_features"]])

        preds = model.predict([lc_local, lc_global, lc_unfolded, centroid, scalar_features])
        probs = preds[:, 1]
        labels = ["Planet Candidate" if p >= 0.5 else "False Positive" for p in probs]

        df["Prediction"] = labels
        df["Confidence"] = (probs * 100).round(2)

        return JSONResponse(
            content={
                "num_records": num_records,
                "predictions": df.to_dict(orient="records"),
            }
        )

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# -------------------------------
# Kepler API endpoint
# -------------------------------
@app.get("/kepid/{kepid}")
def fetch_kepid(kepid: int):
    try:
        data = get_kepid_classification(kepid)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# -------------------------------
# Root route
# -------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the ExoMiner Classification API 🚀"}
