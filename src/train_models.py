"""
train_models.py
----------------
Trains the two models that power PulseCheck AI:

  1. RiskModel     - RandomForestRegressor predicting Risk_Score (0-10)
                      from raw check-in inputs.
  2. PersonaModel  - KMeans clustering that groups people into one of four
                      interpretable "wellness personas".

Artifacts are written to /models as joblib files, plus a metrics.json
report used in the README / dashboard.

Run:
    python src/train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_processing import build_processed_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Age",
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)",
    "Happiness_Index(1-10)",
]

CLUSTER_COLS = [
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Exercise_Frequency(week)",
    "Happiness_Index(1-10)",
]

PERSONA_LIBRARY = {
    # ordered worst -> best mean risk, filled in dynamically below
}


def train_risk_model(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS]
    y = df["Risk_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(model, MODELS_DIR / "risk_model.pkl")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "risk_model_features.pkl")
    return metrics


def name_personas(df: pd.DataFrame, cluster_col: str = "Persona_Cluster") -> dict:
    """Rank clusters by mean risk score and assign descriptive labels."""
    ranked = (
        df.groupby(cluster_col)["Risk_Score"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    labels_low_to_high_risk = [
        "Balanced Thriver",
        "Quietly Drifting",
        "Wired & Tired",
        "Overloaded & At-Risk",
    ]

    # Handle any k != 4 gracefully
    n = len(ranked)
    if n <= len(labels_low_to_high_risk):
        chosen = labels_low_to_high_risk[:n]
    else:
        chosen = labels_low_to_high_risk + [f"Cluster {i}" for i in range(n - len(labels_low_to_high_risk))]

    return {cluster_id: chosen[i] for i, cluster_id in enumerate(ranked)}


def train_persona_model(df: pd.DataFrame, k: int = 4) -> dict:
    X = df[CLUSTER_COLS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["Persona_Cluster"] = clusters
    # temp risk score merge just for naming (Risk_Score already in df)
    persona_names = name_personas(df)

    joblib.dump(kmeans, MODELS_DIR / "persona_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "persona_scaler.pkl")
    joblib.dump(CLUSTER_COLS, MODELS_DIR / "persona_model_features.pkl")
    joblib.dump(persona_names, MODELS_DIR / "persona_names.pkl")

    centroid_summary = (
        df.groupby("Persona_Cluster")[CLUSTER_COLS + ["Risk_Score"]]
        .mean()
        .round(2)
        .rename(index=persona_names)
        .to_dict(orient="index")
    )

    return {
        "k": k,
        "persona_names": {int(k): v for k, v in persona_names.items()},
        "centroid_summary": centroid_summary,
    }


def main():
    df = build_processed_dataset()

    risk_metrics = train_risk_model(df)
    persona_metrics = train_persona_model(df)

    report = {"risk_model": risk_metrics, "persona_model": persona_metrics}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
