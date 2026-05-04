# FinGuard — AI-Driven Real-Time Expense Anomaly Detection

---

## Project Overview

FinGuard detects suspicious financial transactions using machine learning.
It assigns each transaction a risk score (0–100) and explains exactly why
it was flagged — in plain English.

Built to simulate enterprise financial workflows similar to SAP systems,
where finance teams process thousands of transactions and need automated
anomaly detection to catch fraud, errors, and policy violations.

---

## Problem Statement

In large organizations, finance teams process thousands of expense
transactions daily. Manual review is impossible at scale. Traditional
rule-based systems (e.g. "flag any transaction > ₹10,000") miss
context-dependent anomalies and generate too many false positives.

**The real problem:** Anomalies are rare, unlabeled, and context-dependent.
A ₹15,000 travel expense at 3 AM is suspicious. The same amount during
business hours from a frequent traveler might not be.

---

## Solution Approach

1. **Unsupervised ML (Isolation Forest):** No labeled fraud data needed.
   The model learns the "normal" distribution from your data and flags
   transactions that are statistically hard to isolate.

2. **Multi-signal Risk Scoring:** The ML score is one input. We combine it
   with amount deviation, unusual timing, and rare category detection to
   produce a composite 0–100 risk score.

3. **Explainability:** Every flagged transaction gets a plain-English
   explanation — not just a score. This is critical for auditors and
   finance teams who need to act on the output.

4. **Performance-first Design:** Model trains once and loads from disk on
   all subsequent requests. No retraining per upload.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python + Flask | Lightweight, easy to extend |
| ML | Scikit-learn Isolation Forest | Best-in-class unsupervised anomaly detection |
| Data | Pandas + NumPy | Vectorized operations, fast on large CSVs |
| Database | SQLite | Zero-config, embeddable, perfect for demos |
| Frontend | HTML + CSS + Jinja2 | No framework bloat, fully explainable |
| Charts | Matplotlib | Generates static PNGs, no JS charting dependency |
| Model Persistence | Joblib | Fast serialization for scikit-learn models |

---

## Architecture

```
expense-anomaly/
├── app.py                  # Entry point: starts Flask, loads model, registers routes
├── requirements.txt
├── data/
│   └── transactions.csv    # Sample dataset for testing
├── models/                 # Auto-created: stores trained model + scaler
│   ├── isolation_forest.pkl
│   └── scaler.pkl
└── app/
    ├── __init__.py
    ├── model.py            # ML logic only (train, load, predict)
    ├── utils.py            # Preprocessing, risk scoring, charts, DB
    ├── routes.py           # Flask routes (upload, score API, stats API)
    └── templates/
        ├── index.html      # Upload page
        └── results.html    # Dashboard with flagged transactions
```

**Separation of concerns:**
- `model.py` knows nothing about Flask or data cleaning
- `utils.py` knows nothing about ML — only features and scores
- `routes.py` orchestrates the pipeline, knows nothing about internals
- `app.py` only wires everything together

---

## How to Run

### 1. Clone / extract the project
```bash
cd expense-anomaly
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

### 6. Upload the sample CSV
Use `data/transactions.csv` for a quick demo with pre-built anomalies.

---

## API Reference

### POST /score — Programmatic batch scoring
```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "TXN999",
        "amount": 22000,
        "category": "Electronics",
        "user_id": "USR001",
        "timestamp": "2024-01-15 02:30:00"
      }
    ]
  }'
```

Response:
```json
{
  "results": [
    {
      "transaction_id": "TXN999",
      "amount": 22000,
      "anomaly_flag": -1,
      "risk_score": 89.5,
      "explanation": "Amount is 6.2x the average | Transaction at unusual hour (2:00, outside 9AM–6PM) | Isolation Forest model identified this as a statistical outlier"
    }
  ],
  "total": 1
}
```

### GET /api/stats — Summary statistics
```bash
curl http://localhost:5000/api/stats
```

---

## How the ML Works (for interviews)

### Isolation Forest
- Builds random decision trees that try to isolate individual data points
- Normal points are deep in the tree (hard to isolate — similar to others)
- Anomalies are near the root (easy to isolate — different from others)
- `decision_function()` returns a score: more negative = more anomalous
- `contamination=0.1` tells the model to treat ~10% of data as anomalies

### Risk Score Formula
```
risk_score = 
  anomaly_component (0–40)   # From Isolation Forest output
  + amount_deviation (0–30)  # Z-score of the transaction amount
  + time_penalty (0–15)      # Outside 9AM–6PM business hours
  + category_penalty (0–15)  # Rare or unknown category
```
Capped at 100. Rounded to 1 decimal.

### Z-score normalization
```
z = (amount - mean) / std_deviation
```
A z-score > 2 means the amount is 2 standard deviations above average.
We use this as an independent signal (not just the ML score).

---

## Real-World Relevance (SAP Systems)

SAP's Financial Accounting (FI) and Controlling (CO) modules process
millions of transactions. FinGuard mirrors how SAP-adjacent systems
add intelligence on top of raw transaction data:

- **Batch upload** mirrors SAP's BAPI/RFC file exports
- **Risk scoring** mirrors SAP GRC (Governance, Risk, Compliance) alerts
- **Explainability** enables auditors to act without ML knowledge
- **REST API** (`/score`) enables integration with SAP via standard HTTP
- **SQLite storage** can be replaced with SAP HANA or any enterprise DB
- **Isolation Forest** is ideal for SAP scenarios where fraud labels don't exist

---

## Interview Talking Points

1. **Why Isolation Forest?**
   "It's unsupervised — you don't need labeled fraud data. It works by
   isolating anomalies with random splits. Anomalies are isolated faster."

2. **Why not retrain per request?**
   "Training is expensive. We train once and load the model from disk.
   This is standard production ML practice — same pattern used in
   real-time fraud detection at banks."

3. **Why four risk components?**
   "ML alone gives false confidence. Combining statistical outlier score
   with domain knowledge (business hours, amount deviation) is more
   robust and explainable to non-technical stakeholders."

4. **How would you scale this?**
   "Replace SQLite with PostgreSQL. Add a message queue (Kafka/Celery)
   for async batch processing. Deploy the Flask app with Gunicorn behind
   Nginx. Retrain the model nightly as a scheduled job."

5. **What's the weakness of this system?**
   "Isolation Forest assumes anomalies are rare. If the training data
   already contains lots of fraud, contamination estimation breaks down.
   Also, concept drift — transaction patterns shift over time, so the
   model needs periodic retraining."
