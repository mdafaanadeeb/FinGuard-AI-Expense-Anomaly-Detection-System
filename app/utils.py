"""
utils.py
--------
Handles:
1. Data preprocessing (cleaning, feature engineering)
2. Risk score calculation (0-100 composite score)
3. Human-readable explanations for flagged transactions
4. Chart generation
5. Database operations
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import io
import base64
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

DB_PATH = "transactions.db"

# Business hours definition (9 AM to 6 PM)
BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18

# Categories seen in baseline/normal data
COMMON_CATEGORIES = {"Travel", "Food", "Office Supplies", "Consulting", "Electronics"}


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw CSV upload.
    
    Steps:
    1. Drop rows with missing critical fields
    2. Parse timestamp → extract hour
    3. Compute z-score normalized amount
    4. Flag if category is unusual
    
    Args:
        df: Raw dataframe from uploaded CSV
        
    Returns:
        Cleaned dataframe with new feature columns
    """
    required_cols = {"transaction_id", "amount", "category", "user_id", "timestamp"}
    
    # Validate required columns exist
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows where amount or timestamp is null
    df = df.dropna(subset=["amount", "timestamp"]).copy()
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    # Extract hour of day (0-23)
    df["hour"] = df["timestamp"].dt.hour
    
    # Z-score normalization: (value - mean) / std
    # Tells us how many standard deviations away from average
    mean_amt = df["amount"].mean()
    std_amt = df["amount"].std()
    df["amount_zscore"] = (df["amount"] - mean_amt) / (std_amt if std_amt > 0 else 1)
    
    # Store mean and std for use in risk scoring
    df["amount_mean"] = mean_amt
    df["amount_std"] = std_amt
    
    # Flag rare/unknown categories
    df["is_rare_category"] = (~df["category"].isin(COMMON_CATEGORIES)).astype(int)
    
    # Flag transactions outside business hours
    df["is_unusual_time"] = (
        (df["hour"] < BUSINESS_HOUR_START) | (df["hour"] >= BUSINESS_HOUR_END)
    ).astype(int)
    
    return df


# ─── RISK SCORING ─────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite risk score (0–100) for each transaction.
    
    Score is built from 4 components:
    
    1. Anomaly score (40 pts max):
       From Isolation Forest decision_function output.
       More negative = more anomalous = higher risk.
    
    2. Amount deviation (30 pts max):
       How far the amount is from the dataset mean (in std devs).
       >3 std devs → max points.
    
    3. Unusual time (15 pts max):
       Transactions outside 9AM–6PM get full points.
    
    4. Rare category (15 pts max):
       Categories not in common set get full points.
    
    Final score = sum of components, capped at 100.
    
    Args:
        df: Dataframe with anomaly_flag, anomaly_score, and feature columns
        
    Returns:
        df with risk_score column added
    """
    df = df.copy()
    
    # Component 1: Anomaly score contribution (0–40)
    # decision_function range is roughly -0.5 to 0.5
    # Clip and normalize to 0–1, then scale to 40
    anomaly_component = np.clip(-df["anomaly_score"], 0, 0.5) / 0.5 * 40
    
    # Component 2: Amount deviation (0–30)
    # Cap at 3 standard deviations for scoring purposes
    amount_component = np.clip(df["amount_zscore"].abs(), 0, 3) / 3 * 30
    
    # Component 3: Unusual time (0–15)
    time_component = df["is_unusual_time"] * 15
    
    # Component 4: Rare category (0–15)
    category_component = df["is_rare_category"] * 15
    
    # Sum and clip to 0–100
    df["risk_score"] = np.clip(
        anomaly_component + amount_component + time_component + category_component,
        0, 100
    ).round(1)
    
    return df


# ─── EXPLAINABILITY ───────────────────────────────────────────────────────────

def generate_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate human-readable reasons for each flagged transaction.
    
    Multiple reasons can apply to a single transaction.
    Reasons are concatenated into a single string.
    
    Args:
        df: Dataframe with risk score and feature columns
        
    Returns:
        df with 'explanation' column
    """
    df = df.copy()
    
    def build_reason(row) -> str:
        reasons = []
        
        # High amount
        if row["amount_zscore"] > 2:
            multiplier = round(row["amount"] / row["amount_mean"], 1)
            reasons.append(f"Amount is {multiplier}x the average ({row['amount_mean']:.0f})")
        
        # Very high amount
        if row["amount_zscore"] > 3:
            reasons.append("Extremely high transaction amount (>3 std deviations from mean)")
        
        # Unusual hour
        if row["is_unusual_time"]:
            reasons.append(f"Transaction at unusual hour ({int(row['hour'])}:00, outside 9AM–6PM)")
        
        # Rare category
        if row["is_rare_category"]:
            reasons.append(f"Rare or unrecognized category: '{row['category']}'")
        
        # Isolation forest flagged
        if row["anomaly_flag"] == -1:
            reasons.append("Isolation Forest model identified this as a statistical outlier")
        
        # Default if no specific reason found
        if not reasons:
            reasons.append("Mild anomaly pattern detected — monitor for recurrence")
        
        return " | ".join(reasons)
    
    # Apply row-wise (unavoidable here since logic is conditional)
    df["explanation"] = df.apply(build_reason, axis=1)
    
    return df


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame) -> str:
    """
    Generate a base64-encoded PNG chart for the dashboard.
    
    Chart shows:
    - All transactions plotted by amount and risk score
    - Anomalies highlighted in red
    
    Returns:
        Base64-encoded PNG string (embeddable in HTML)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Plot 1: Amount vs Risk Score scatter
    normal = df[df["anomaly_flag"] == 1]
    anomalies = df[df["anomaly_flag"] == -1]
    
    axes[0].scatter(normal["amount"], normal["risk_score"],
                    c="#22d3ee", alpha=0.7, s=60, label="Normal", zorder=3)
    axes[0].scatter(anomalies["amount"], anomalies["risk_score"],
                    c="#f87171", alpha=0.9, s=90, marker="^", label="Anomaly", zorder=4)
    axes[0].set_xlabel("Transaction Amount", color="#94a3b8", fontsize=11)
    axes[0].set_ylabel("Risk Score", color="#94a3b8", fontsize=11)
    axes[0].set_title("Amount vs Risk Score", color="#e2e8f0", fontsize=13, pad=15)
    axes[0].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[0].grid(True, color="#334155", alpha=0.5)
    
    # Plot 2: Risk score distribution
    axes[1].hist(df["risk_score"], bins=20, color="#6366f1", alpha=0.8, edgecolor="#334155")
    axes[1].axvline(x=50, color="#f87171", linestyle="--", linewidth=1.5, label="Risk threshold (50)")
    axes[1].set_xlabel("Risk Score", color="#94a3b8", fontsize=11)
    axes[1].set_ylabel("Count", color="#94a3b8", fontsize=11)
    axes[1].set_title("Risk Score Distribution", color="#e2e8f0", fontsize=13, pad=15)
    axes[1].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[1].grid(True, color="#334155", alpha=0.5)
    
    plt.tight_layout(pad=2.0)
    
    # Convert to base64 for embedding in HTML
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded


# ─── DATABASE ─────────────────────────────────────────────────────────────────

def init_db():
    """Create the SQLite database and transactions table if not present."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            category TEXT,
            user_id TEXT,
            timestamp TEXT,
            hour INTEGER,
            amount_zscore REAL,
            anomaly_flag INTEGER,
            anomaly_score REAL,
            risk_score REAL,
            explanation TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(df: pd.DataFrame):
    """
    Insert processed transactions into SQLite.
    
    Uses pandas to_sql with append mode for efficiency.
    Clears old records first to avoid duplicates between uploads.
    """
    conn = sqlite3.connect(DB_PATH)
    
    cols = [
        "transaction_id", "amount", "category", "user_id", "timestamp",
        "hour", "amount_zscore", "anomaly_flag", "anomaly_score",
        "risk_score", "explanation"
    ]
    
    # Only keep columns that exist
    save_cols = [c for c in cols if c in df.columns]
    df_save = df[save_cols].copy()
    df_save["timestamp"] = df_save["timestamp"].astype(str)
    
    # Clear previous batch and insert fresh
    conn.execute("DELETE FROM transactions")
    df_save.to_sql("transactions", conn, if_exists="append", index=False)
    
    conn.commit()
    conn.close()
