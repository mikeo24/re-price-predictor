import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_pipeline_compressed.joblib")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "model", "market_snapshot_by_zip.csv")

model = joblib.load(MODEL_PATH)
market_snapshot = pd.read_csv(SNAPSHOT_PATH, dtype={"zip_code": str})
market_snapshot = market_snapshot.set_index("zip_code")


MODEL_MEDAPE = 0.123

# Column groups the pipeline expects, in the raw (unlogged) units it was trained on
REQUIRED_PROPERTY_FIELDS = ["zip_code", "bed", "bath", "house_size", "acre_lot"]

app = Flask("price_prediction")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    # --- validate required fields ---
    missing = [f for f in REQUIRED_PROPERTY_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    zip_code = str(data["zip_code"]).zfill(5)

    # --- look up market context for this ZIP ---
    if zip_code not in market_snapshot.index:
        return jsonify({"error": f"No market data available for ZIP {zip_code}."}), 404

    market_row = market_snapshot.loc[zip_code]

    # --- assemble the one-row input the pipeline expects ---
    try:
        row = {
            "house_size": float(data["house_size"]),
            "zhvi": float(market_row["zhvi"]),
            "median_ppsf": float(market_row["median_ppsf"]),
            "acre_lot": float(data["acre_lot"]),
            "active_listing_count": float(market_row["active_listing_count"]),
            "homes_sold": float(market_row["homes_sold"]),
            "median_days_on_market": float(market_row["median_days_on_market"]),
            "bed": float(data["bed"]),
            "bath": float(data["bath"]),
            "avg_sale_to_list": float(market_row["avg_sale_to_list"]),
            "sold_above_list": float(market_row["sold_above_list"]),
            "price_reduced_share": float(market_row["price_reduced_share"]),
            "sale_month": int(data.get("sale_month", datetime.now().month)),
            "state": market_row["state"],
            "parent_metro_region": market_row["parent_metro_region"],
            "broker_cat": "other",  # not knowable from the user's inputs
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input value: {e}"}), 400

    X = pd.DataFrame([row])

    # --- predict (pipeline returns real dollars directly) ---
    price = float(model.predict(X)[0])
    low = price * (1 - MODEL_MEDAPE)
    high = price * (1 + MODEL_MEDAPE)

    return jsonify({
        "estimate": round(price),
        "range_low": round(low),
        "range_high": round(high),
        "market_context": {
            "zip_code": zip_code,
            "typical_home_value": round(float(market_row["zhvi"])),
            "price_per_sqft": round(float(market_row["median_ppsf"])),
            "median_days_on_market": round(float(market_row["median_days_on_market"])),
            "sold_above_list_pct": round(float(market_row["sold_above_list"]) * 100, 1),
        },
    })


if __name__ == "__main__":
    app.run(debug=True)