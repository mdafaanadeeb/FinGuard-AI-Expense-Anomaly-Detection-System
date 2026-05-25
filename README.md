# 🚀 FinGuard AI – Real-Time Expense Anomaly Detection System

---

## 🌍 Live Demo

👉 https://finguard-anomaly-detector.onrender.com/

---

## 📌 Overview

**FinGuard AI** is an AI-powered financial monitoring system that detects suspicious transactions using machine learning.

It analyzes transaction data, identifies anomalies, assigns a **risk score (0–100)**, and provides **clear explanations** for each flagged transaction.

This project is designed to simulate **real-world enterprise financial workflows**, similar to systems used alongside SAP.

---

## 🎯 Problem Statement

In large organizations:

* Thousands of financial transactions occur daily
* Manual auditing is slow and inefficient
* Rule-based systems fail to detect complex anomalies

👉 **Challenge:** Detect unusual transactions without labeled fraud data

---

## 💡 Solution

FinGuard AI uses **unsupervised machine learning (Isolation Forest)** to identify anomalies and combines it with business rules to produce a **reliable, explainable risk score**.

---

## 🧠 Key Features

* 📊 Upload transaction dataset (CSV)
* 🤖 AI-based anomaly detection (Isolation Forest)
* 📈 Risk scoring system (0–100 scale)
* 🧾 Human-readable explanations for each anomaly
* ⚡ Fast processing (model loaded once, no retraining per request)
* 🌐 Web-based interface using Flask
* 🔌 API endpoints for integration with external systems

---

## ⚙️ Tech Stack

| Layer            | Technology                      |
| ---------------- | ------------------------------- |
| Backend          | Python, Flask                   |
| Machine Learning | Scikit-learn (Isolation Forest) |
| Data Processing  | Pandas, NumPy                   |
| Visualization    | Matplotlib                      |
| Database         | SQLite                          |
| Deployment       | Gunicorn, Render                |
| Frontend         | HTML, CSS, Jinja2               |

---

## 🔁 How It Works

1. Upload CSV file
2. Data preprocessing (cleaning + feature extraction)
3. Model detects anomalies using Isolation Forest
4. Risk score is calculated using multiple signals:

   * Amount deviation
   * Transaction timing
   * Category behavior
   * ML anomaly score
5. Results displayed with explanations

---

## 📊 Risk Score Logic

```
Risk Score =
  ML anomaly score +
  Amount deviation +
  Time-based anomaly +
  Category-based anomaly
```

👉 Output: **0–100 risk score + explanation**

---

## 🌍 Real-World Application

* Financial fraud detection
* Expense monitoring systems
* SAP Finance (FI) integration layer
* Audit and compliance automation
* Banking and fintech analytics

---
