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
    DIM_WEIGHTS, compute_sub_scores, compute_overall
)
 
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
 
GMAPS_KEY = ""
gmaps     = googlemaps.Client(GMAPS_KEY)
RADIUS    = 1600
 
scaler = joblib.load("scaler.pkl")
model  = LivabilityModel(input_size=len(FEATURE_COLS))
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()
 
df_cache               = pd.read_csv("campus_features.csv")
df_cache["name_lower"] = df_cache["name"].str.lower().str.strip()
 
live_cache: dict = {}
 
DIMS = ["walkability", "food_amenities", "campus_spirit", "weather", "safety"]
 
# Default weights used when no preferences are supplied.
# These reflect a balanced livability + walkability baseline.
DEFAULT_WEIGHTS = {
    "walkability":    0.30,   # boosted — core livability signal
    "food_amenities": 0.20,
    "campus_spirit":  0.15,
    "weather":        0.20,
    "safety":         0.15,
}
 
 
def parse_weights(args) -> dict:
    """
    Build a normalised weight dict from query params.
    Any dimension not supplied falls back to DEFAULT_WEIGHTS.
    Weights are normalised server-side so the caller never has to sum to 1.
    """
    weights = {}
    for dim in DIMS:
        val = args.get(dim)
        if val is not None:
            try:
                weights[dim] = max(0.0, float(val))   # clamp negatives to 0
            except ValueError:
                weights[dim] = DEFAULT_WEIGHTS[dim]
        else:
            weights[dim] = DEFAULT_WEIGHTS[dim]
 
    total = sum(weights.values())
    if total == 0:
        return DEFAULT_WEIGHTS.copy()
 
    return {k: v / total for k, v in weights.items()}
 
 
def score_row(row: pd.Series, custom_weights: dict = None) -> dict:
    """
    Score a single campus row.
    formula_score uses custom_weights (or DEFAULT_WEIGHTS).
    nn_score reflects the model's fixed learned weighting and is blended in at 40%.
    """
    sub     = compute_sub_scores(row)
    weights = custom_weights or DEFAULT_WEIGHTS
    overall = compute_overall(sub, weights)
 
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
        "formula_score": round(overall, 1),
        "nn_score":      round(nn_score, 1),
        "sub_scores":    sub,
        "weights_used":  weights,
        "source":        row.get("source", "dataset"),
    }
 
 
def fetch_live(name: str, custom_weights: dict = None) -> dict:
    """Geocode + fetch live POI/weather data for a campus not in the dataset."""
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
    result = score_row(row, custom_weights)
    live_cache[name.lower()] = result
    return result
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.route("/")
def index():
    return render_template("index.html", maps_key=GMAPS_KEY)
 
 
@app.route("/api/score")
def api_score():
    """
    Score a single campus, optionally with custom dimension weights.
 
    Query params:
        campus      (required) campus name
        walkability, food_amenities, campus_spirit, weather, safety
                    (optional) relative preference weights — any positive numbers,
                    they are normalised server-side.
 
    Example:
        /api/score?campus=MIT&weather=3&walkability=2
    """
    name = request.args.get("campus", "").strip()
    if not name:
        return jsonify({"error": "campus parameter required"}), 400
 
    custom_weights = parse_weights(request.args)
    key = name.lower()
 
    if key in live_cache:
        cached = live_cache[key]
        row = pd.Series({
            "name": cached["campus"],
            "lat":  cached["lat"],
            "lng":  cached["lng"],
            **cached.get("sub_scores", {}),
            "source": "live_cache",
        })
        return jsonify({**score_row(row, custom_weights), "source": "live_cache"})
 
    match = df_cache[df_cache["name_lower"].str.contains(key, na=False)]
    if not match.empty:
        row = match.iloc[0].copy()
        row["source"] = "dataset"
        return jsonify(score_row(row, custom_weights))
 
    return jsonify(fetch_live(name, custom_weights))
 
 
@app.route("/api/rank")
def api_rank():
    """
    Rank all dataset campuses under the given preference weights.
    Returns campuses sorted highest overall_score first.
 
    Query params (all optional):
        walkability, food_amenities, campus_spirit, weather, safety
        limit   — max results to return (default: all)
        top     — alias for limit
 
    Example:
        /api/rank?weather=3&walkability=2&limit=10
    """
    custom_weights = parse_weights(request.args)
 
    try:
        limit = int(request.args.get("limit") or request.args.get("top") or 0)
    except ValueError:
        limit = 0
 
    results = []
    for _, row in df_cache.iterrows():
        results.append(score_row(row, custom_weights))
 
    results.sort(key=lambda x: x["overall_score"], reverse=True)
 
    if limit > 0:
        results = results[:limit]
 
    return jsonify({
        "weights_used": custom_weights,
        "count":        len(results),
        "campuses":     results,
    })
 
 
@app.route("/api/campuses")
def api_campuses():
    """Return the list of all campus names in the dataset."""
    return jsonify(df_cache["name"].tolist())
 
 
@app.route("/api/weights/default")
def api_default_weights():
    """Return the default dimension weights so the frontend can initialise sliders."""
    return jsonify(DEFAULT_WEIGHTS)
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)