"""
main.py - PulseCheck AI backend

FastAPI app exposing:
  GET  /api/health
  POST /api/predict         -> risk score + persona + recommendations for one-shot form input
  POST /api/chat/start       -> begin a guided conversational check-in
  POST /api/chat/message     -> continue a guided conversational check-in

Also serves the static frontend from /frontend at "/".
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from recommender import CheckInInput, build_recommendation  # noqa: E402
from chatbot import start_session, handle_message  # noqa: E402
from data_processing import compute_risk_score  # noqa: E402
from llm_explainer import explain  # noqa: E402

MODELS_DIR = ROOT / "models"

app = FastAPI(title="PulseCheck AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------------
risk_model = joblib.load(MODELS_DIR / "risk_model.pkl")
risk_features = joblib.load(MODELS_DIR / "risk_model_features.pkl")

persona_model = joblib.load(MODELS_DIR / "persona_model.pkl")
persona_scaler = joblib.load(MODELS_DIR / "persona_scaler.pkl")
persona_features = joblib.load(MODELS_DIR / "persona_model_features.pkl")
persona_names = joblib.load(MODELS_DIR / "persona_names.pkl")


def _to_feature_row(inp: CheckInInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "Age": inp.age,
        "Daily_Screen_Time(hrs)": inp.screen_time_hrs,
        "Sleep_Quality(1-10)": inp.sleep_quality,
        "Stress_Level(1-10)": inp.stress_level,
        "Days_Without_Social_Media": inp.days_without_social_media,
        "Exercise_Frequency(week)": inp.exercise_frequency,
        "Happiness_Index(1-10)": inp.happiness_index,
    }])


def predict(inp: CheckInInput):
    row = _to_feature_row(inp)

    # Served / escalation-critical score: the transparent, auditable formula.
    risk_score = compute_risk_score(
        stress_level=inp.stress_level,
        sleep_quality=inp.sleep_quality,
        screen_time_hrs=inp.screen_time_hrs,
        happiness_index=inp.happiness_index,
        exercise_frequency=inp.exercise_frequency,
        days_without_social_media=inp.days_without_social_media,
    )

    # RF model's independent estimate, kept for model-validation / transparency
    # (e.g. surfaced in the "model" section of the UI) - not used for escalation.
    model_estimate = float(risk_model.predict(row[risk_features])[0])
    model_estimate = max(0.0, min(10.0, round(model_estimate, 2)))

    cluster_row = row[persona_features]
    scaled = persona_scaler.transform(cluster_row)
    cluster_id = int(persona_model.predict(scaled)[0])
    persona = persona_names[cluster_id]

    return risk_score, persona, model_estimate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def api_predict(payload: dict):
    inp = CheckInInput(
        age=payload["age"],
        screen_time_hrs=payload["screen_time_hrs"],
        sleep_quality=payload["sleep_quality"],
        stress_level=payload["stress_level"],
        days_without_social_media=payload["days_without_social_media"],
        exercise_frequency=payload["exercise_frequency"],
        happiness_index=payload["happiness_index"],
    )
    risk_score, persona, model_estimate = predict(inp)
    rec = build_recommendation(inp, risk_score, persona)

    ai_explanation = explain(
        risk_score=rec.risk_score,
        risk_tier=rec.risk_tier,
        persona=rec.persona,
        stress_level=inp.stress_level,
        sleep_quality=inp.sleep_quality,
        screen_time_hrs=inp.screen_time_hrs,
        happiness_index=inp.happiness_index,
        exercise_frequency=inp.exercise_frequency,
        days_without_social_media=inp.days_without_social_media,
    )

    return {
        "risk_score": rec.risk_score,
        "risk_tier": rec.risk_tier,
        "persona": rec.persona,
        "headline": rec.headline,
        "escalate": rec.escalate,
        "tips": rec.tips,
        "crisis_resources": rec.crisis_resources,
        "model_estimate": model_estimate,
        "ai_explanation": ai_explanation,
    }


@app.post("/api/chat/start")
def api_chat_start(payload: dict):
    session_id = payload["session_id"]
    return start_session(session_id)


def _chat_predictor(inp: CheckInInput):
    risk_score, persona, _ = predict(inp)
    return risk_score, persona


@app.post("/api/chat/message")
def api_chat_message(payload: dict):
    session_id = payload["session_id"]
    message = payload["message"]
    return handle_message(session_id, message, _chat_predictor, explain)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
FRONTEND_DIR = ROOT / "frontend"


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")
