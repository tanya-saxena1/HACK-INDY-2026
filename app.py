import os, time, requests
import googlemaps
import joblib
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import (
    LivabilityModel, FEATURE_COLS, POI_TYPES,
    compute_sub_scores, compute_overall
)

app     = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

GMAPS_KEY = ""
gmaps     = googlemaps.Client(GMAPS_KEY)
RADIUS    = 1600

scaler = joblib.load("scaler.pkl")
model  = LivabilityModel(input_size=len(FEATURE_COLS))
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

df_cache              = pd.read_csv("campus_features.csv")
df_cache["name_lower"] = df_cache["name"].str.lower().str.strip()

live_cache: dict = {}


def score_row(row: pd.Series) -> dict:
    sub     = compute_sub_scores(row)
    overall = compute_overall(sub)

    feats    = np.array([[row.get(c, 0) or 0 for c in FEATURE_COLS]], dtype=np.float32)
    feats_sc = scaler.transform(feats)
    with torch.no_grad():
        nn_score = float(model(torch.FloatTensor(feats_sc)).item())

    blended = round(overall * 0.6 + nn_score * 0.4, 1)

    return {
        "campus":        row.get("name", "Unknown"),
        "lat":           float(row["lat"]),
        "lng":           float(row["lng"]),
        "overall_score": blended,
        "formula_score": overall,
        "nn_score":      round(nn_score, 1),
        "sub_scores":    sub,
        "source":        row.get("source", "dataset"),
    }


def fetch_live(name: str) -> dict:
    results = gmaps.geocode(f"{name} university campus")
    if not results:
        return {"error": f"Could not locate: {name}"}
    loc = results[0]["geometry"]["location"]
    lat, lng = loc["lat"], loc["lng"]

    poi = {}
    for label, ptype in POI_TYPES.items():
        try:
            r = gmaps.places_nearby(location=(lat, lng), radius=RADIUS, type=ptype)
            poi[label] = len(r.get("results", []))
            time.sleep(0.1)
        except Exception:
            poi[label] = 0

    weather = {"avg_temp_c": 15.0, "avg_precip_mm": 3.0, "avg_sun_hrs": 6.0}
    try:
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration"
            "&timezone=auto&forecast_days=7",
            timeout=6
        ).json().get("daily", {})
        weather = {
            "avg_temp_c":    round((sum(w["temperature_2m_max"]) + sum(w["temperature_2m_min"])) / 14, 1),
            "avg_precip_mm": round(sum(w["precipitation_sum"]) / 7, 2),
            "avg_sun_hrs":   round(sum(w["sunshine_duration"]) / 7 / 3600, 1),
        }
    except Exception:
        pass

    row    = pd.Series({"name": name, "lat": lat, "lng": lng, **poi, **weather, "source": "live"})
    result = score_row(row)
    live_cache[name.lower()] = result
    return result


@app.route("/")
def index():
    return render_template("index.html", maps_key=GMAPS_KEY)


@app.route("/api/score")
def api_score():
    name = request.args.get("campus", "").strip()
    if not name:
        return jsonify({"error": "campus parameter required"}), 400

    key = name.lower()

    if key in live_cache:
        return jsonify({**live_cache[key], "source": "live_cache"})

    match = df_cache[df_cache["name_lower"].str.contains(key, na=False)]
    if not match.empty:
        row = match.iloc[0].copy()
        row["source"] = "dataset"
        return jsonify(score_row(row))

    return jsonify(fetch_live(name))


@app.route("/api/campuses")
def api_campuses():
    return jsonify(df_cache["name"].tolist())


if __name__ == "__main__":
    app.run(debug=True, port=5000)