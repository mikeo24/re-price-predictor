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

# Market-side features looked up from the ZIP snapshot. Roughly a quarter of ZIPs
# have gaps in some of these (Redfin/Zillow don't cover every ZIP), so we track
# which ones are missing instead of letting a NaN crash the request.
MARKET_NUMERIC_FIELDS = [
    "zhvi", "median_ppsf", "active_listing_count", "homes_sold",
    "median_days_on_market", "avg_sale_to_list", "sold_above_list",
    "price_reduced_share",
]
MARKET_FIELDS = MARKET_NUMERIC_FIELDS + ["parent_metro_region"]

app = Flask("price_prediction")


def _clean(value, ndigits=None, scale=1.0):
    """Return a JSON-safe number, or None if the value is missing/NaN."""
    if value is None or pd.isna(value):
        return None
    out = float(value) * scale
    return round(out, ndigits) if ndigits is not None else round(out)


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
    missing_market = [f for f in MARKET_FIELDS if pd.isna(market_row[f])]

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
    try:
        price = float(model.predict(X)[0])
        if not np.isfinite(price):
            raise ValueError("model returned a non-finite prediction")
    except Exception:
        app.logger.exception("Prediction failed for ZIP %s (missing market fields: %s)",
                             zip_code, missing_market)
        if missing_market:
            return jsonify({
                "error": (f"Market data for ZIP {zip_code} is incomplete "
                          f"({len(missing_market)} of {len(MARKET_FIELDS)} market signals unavailable), "
                          "so a reliable estimate can't be produced for this ZIP."),
                "missing_market_fields": missing_market,
            }), 422
        return jsonify({"error": "The model couldn't produce an estimate for these inputs."}), 500

    low = price * (1 - MODEL_MEDAPE)
    high = price * (1 + MODEL_MEDAPE)

    return jsonify({
        "estimate": round(price),
        "range_low": round(low),
        "range_high": round(high),
        "missing_market_fields": missing_market,
        "market_context": {
            "zip_code": zip_code,
            "typical_home_value": _clean(market_row["zhvi"]),
            "price_per_sqft": _clean(market_row["median_ppsf"]),
            "median_days_on_market": _clean(market_row["median_days_on_market"]),
            "sold_above_list_pct": _clean(market_row["sold_above_list"], ndigits=1, scale=100),
        },
    })


if __name__ == "__main__":
    app.run(debug=True)