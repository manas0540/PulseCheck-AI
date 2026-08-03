"""
recommender.py
---------------
Rule-based recommendation + escalation engine.

Given a user's raw check-in inputs plus the ML outputs (risk score,
persona), this module decides which actionable tips to surface and
whether the "serious stage" escalation banner (Risk_Score > 8.5) should
fire, per the platform's original design brief.

No network calls, no external services - fully self-contained so the
repo runs offline out of the box.
"""

from dataclasses import dataclass, field
from typing import List


TIP_BANK = {
    "breathing": [
        "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat for 2 minutes.",
        "Do 5 rounds of 4-7-8 breathing: in for 4, hold for 7, out for 8.",
    ],
    "digital_detox": [
        "Set a 90-minute phone-free block this evening and put the phone in another room.",
        "Turn off non-essential notifications for the next 3 hours.",
        "Try a 1-hour screen break right after this check-in - go for a short walk instead.",
    ],
    "sleep": [
        "Aim to be in bed 30 minutes earlier tonight - even one extra sleep cycle helps recovery.",
        "Avoid screens for 30 minutes before bed; dim lights instead of scrolling.",
    ],
    "movement": [
        "A brisk 10-minute walk can measurably lower cortisol - even a short one counts.",
        "Try 5 minutes of stretching or light movement between tasks today.",
    ],
    "break": [
        "Take a proper 15-minute break away from your screen before your next task.",
        "Step outside for a few minutes of natural light - it resets attention and mood.",
    ],
    "connection": [
        "Reach out to one person today just to check in - social contact is protective against burnout.",
    ],
}

CRISIS_RESOURCES = [
    {"region": "United States", "name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988"},
    {"region": "United Kingdom & ROI", "name": "Samaritans", "contact": "Call 116 123"},
    {"region": "India", "name": "Tele-MANAS", "contact": "Call 14416"},
    {"region": "Elsewhere", "name": "findahelpline.com", "contact": "Search your country for a free, confidential local helpline"},
]

ESCALATION_THRESHOLD = 8.5


@dataclass
class CheckInInput:
    age: int
    screen_time_hrs: float
    sleep_quality: float
    stress_level: float
    days_without_social_media: float
    exercise_frequency: float
    happiness_index: float


@dataclass
class Recommendation:
    risk_score: float
    risk_tier: str
    persona: str
    escalate: bool
    tips: List[str] = field(default_factory=list)
    crisis_resources: List[dict] = field(default_factory=list)
    headline: str = ""


def _risk_tier(score: float) -> str:
    if score <= 3.5:
        return "Low"
    if score <= 6.0:
        return "Moderate"
    if score <= 8.5:
        return "Elevated"
    return "Critical"


def build_recommendation(inputs: CheckInInput, risk_score: float, persona: str) -> Recommendation:
    tier = _risk_tier(risk_score)
    escalate = risk_score > ESCALATION_THRESHOLD

    tips: List[str] = []

    # Prioritise the single biggest driver of risk for this specific person
    if inputs.sleep_quality <= 5:
        tips.append(TIP_BANK["sleep"][0])
    if inputs.screen_time_hrs >= 6:
        tips.append(TIP_BANK["digital_detox"][0])
    if inputs.stress_level >= 7:
        tips.append(TIP_BANK["breathing"][0])
    if inputs.exercise_frequency <= 1:
        tips.append(TIP_BANK["movement"][0])
    if not tips:
        # Low-risk / already-balanced people still get one light-touch nudge
        tips.append(TIP_BANK["connection"][0])
    tips.append(TIP_BANK["break"][0])

    # De-duplicate while preserving order, cap at 4 tips so it stays actionable
    seen = set()
    deduped = []
    for t in tips:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    tips = deduped[:4]

    headlines = {
        "Low": "You're in a good place right now. Here's how to protect it.",
        "Moderate": "A few small adjustments could meaningfully lower your load.",
        "Elevated": "Your check-in shows real strain. These steps can help today.",
        "Critical": "Your check-in indicates a serious level of strain right now.",
    }

    return Recommendation(
        risk_score=risk_score,
        risk_tier=tier,
        persona=persona,
        escalate=escalate,
        tips=tips,
        crisis_resources=CRISIS_RESOURCES if escalate else [],
        headline=headlines[tier],
    )
