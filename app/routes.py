"""
routes.py
---------
Flask route definitions.

Endpoints:
- GET  /          → Dashboard (upload page)
- POST /upload    → Process uploaded CSV
- GET  /results   → Show results table + chart
- POST /score     → API endpoint for single/batch JSON scoring
- GET  /api/stats → JSON summary stats
"""

import io
import json
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session

from app.model import train_model, load_model, predict
from app.utils import (
    preprocess, compute_risk_score, generate_explanations,
    generate_chart, save_to_db
)

# Blueprint groups all routes under one object
# app.py registers this blueprint
bp = Blueprint("main", __name__)

# Global model state (loaded once at startup, reused per request)
# This avoids the expensive retraining on every request
_model = None
_scaler = None


def get_model(df=None):
    """
    Return cached model. If no model exists, train a new one.
    
    Pattern: Load from disk first (fastest). If not found,
    train on the uploaded data and save. All subsequent calls
    use the in-memory cached version.
    """
    global _model, _scaler
    
    if _model is None:
        _model, _scaler = load_model()
    
    if _model is None and df is not None:
        _model, _scaler = train_model(df)
    
    return _model, _scaler


@bp.route("/")
def index():
    """Render the upload dashboard page."""
    return render_template("index.html")


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Handle CSV file upload.
    
    Steps:
    1. Read uploaded file into dataframe
    2. Preprocess (clean + feature engineering)
    3. Train/load model
    4. Predict anomalies
    5. Compute risk scores
    6. Generate explanations
    7. Save to DB
    8. Store results in session for results page
    """
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")
    
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return render_template("index.html", error="Only CSV files are supported")
    
    try:
        # Read CSV into dataframe
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        
        # Preprocess
        df = preprocess(df)
        
        # Get or train model
        model, scaler = get_model(df)
        
        # Predict anomalies
        df = predict(df, model, scaler)
        
        # Compute risk scores
        df = compute_risk_score(df)
        
        # Generate explanations
        df = generate_explanations(df)
        
        # Save to database
        save_to_db(df)
        
        # Generate chart
        chart_b64 = generate_chart(df)
        
        # Prepare summary stats
        total = len(df)
        anomaly_count = int((df["anomaly_flag"] == -1).sum())
        high_risk_count = int((df["risk_score"] >= 70).sum())
        avg_amount = round(df["amount"].mean(), 2)
        
        # Build flagged transactions table
        flagged = df[df["anomaly_flag"] == -1][[
            "transaction_id", "amount", "category", "user_id",
            "hour", "risk_score", "explanation"
        ]].sort_values("risk_score", ascending=False)
        
        flagged_records = flagged.to_dict(orient="records")
        
        return render_template(
            "results.html",
            total=total,
            anomaly_count=anomaly_count,
            high_risk_count=high_risk_count,
            avg_amount=avg_amount,
            flagged=flagged_records,
            chart_b64=chart_b64
        )
    
    except ValueError as e:
        return render_template("index.html", error=str(e))
    except Exception as e:
        return render_template("index.html", error=f"Processing error: {str(e)}")


@bp.route("/score", methods=["POST"])
def score():
    """
    API endpoint for scoring single or batch transactions via JSON.
    
    Request format:
    {
        "transactions": [
            {
                "transaction_id": "TXN001",
                "amount": 5000,
                "category": "Travel",
                "user_id": "USR001",
                "timestamp": "2024-01-15 03:00:00"
            }
        ]
    }
    
    Response format:
    {
        "results": [
            {
                "transaction_id": "TXN001",
                "anomaly_flag": -1,
                "risk_score": 78.5,
                "explanation": "..."
            }
        ]
    }
    """
    data = request.get_json()
    
    if not data or "transactions" not in data:
        return jsonify({"error": "Request must contain 'transactions' array"}), 400
    
    try:
        df = pd.DataFrame(data["transactions"])
        df = preprocess(df)
        
        model, scaler = get_model(df)
        df = predict(df, model, scaler)
        df = compute_risk_score(df)
        df = generate_explanations(df)
        
        result_cols = ["transaction_id", "amount", "category",
                       "anomaly_flag", "risk_score", "explanation"]
        result_cols = [c for c in result_cols if c in df.columns]
        
        results = df[result_cols].to_dict(orient="records")
        return jsonify({"results": results, "total": len(results)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/stats")
def api_stats():
    """
    Return JSON summary of latest batch results from the database.
    Useful for integration with other systems.
    """
    import sqlite3
    try:
        conn = sqlite3.connect("transactions.db")
        df = pd.read_sql("SELECT * FROM transactions", conn)
        conn.close()
        
        if df.empty:
            return jsonify({"message": "No data processed yet"}), 404
        
        stats = {
            "total_transactions": len(df),
            "anomalies_detected": int((df["anomaly_flag"] == -1).sum()),
            "high_risk_count": int((df["risk_score"] >= 70).sum()),
            "avg_risk_score": round(df["risk_score"].mean(), 2),
            "max_risk_score": round(df["risk_score"].max(), 2),
            "avg_amount": round(df["amount"].mean(), 2),
            "max_amount": round(df["amount"].max(), 2),
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
