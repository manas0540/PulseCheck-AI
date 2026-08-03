"""
llm_explainer.py
-----------------
Optional natural-language explanation layer (feature 7 from the project
brief). Turns the deterministic risk breakdown into a short, human-readable
explanation of *why* the score is what it is and *what specifically* would
move it - e.g.

    "Your stress appears elevated mainly due to reduced sleep and increased
    screen time. Improving sleep consistency by 30-45 minutes could
    meaningfully reduce your predicted risk."

Design contract:
  - This layer NEVER changes the risk score, tier, persona, or the
    crisis-escalation decision. It only narrates numbers already computed
    by data_processing.compute_risk_score / decompose_risk.
  - It is additive and optional: with no API key set, the repo still runs
    fully offline using a deterministic template that follows the exact
    same phrasing pattern as the brief's example.
  - With GROQ_API_KEY or GEMINI_API_KEY set, it calls a free-tier LLM for a
    slightly more fluent, varied explanation grounded in the same numbers.
    Groq is tried first (fast, generous free tier, OpenAI-compatible),
    then Gemini. Any failure (missing key, no package, network, rate
    limit) silently falls back to the template - it never breaks the
    check-in flow. Both are free to sign up for, which matters for a repo
    meant to be cloned and run by anyone without a billing setup.
"""

import os
from typing import Optional

from data_processing import decompose_risk

DRIVER_LABELS = {
    "stress": "elevated stress",
    "sleep": "reduced sleep quality",
    "screen_time": "increased screen time",
    "happiness": "a lower overall mood",
}

IMPROVEMENT_HINTS = {
    "stress": "Bringing your stress down even slightly, e.g. with a short daily breathing practice,",
    "sleep": "Improving your sleep consistency by roughly 30-45 minutes a night",
    "screen_time": "Cutting daily screen time by an hour or two",
    "happiness": "Small mood-lifting habits like movement, daylight, or social contact",
}


DRIVER_THRESHOLD = 1.2  # a factor must contribute at least this much to be called out by name


def _fallback_explanation(risk_score: float, risk_tier: str, contributions: dict) -> str:
    positive = {
        k: v for k, v in contributions.items()
        if k in DRIVER_LABELS and v >= DRIVER_THRESHOLD
    }

    if risk_tier == "Low" or not positive:
        return (
            f"Your check-in comes back {risk_tier.lower()} risk ({risk_score}/10) — "
            "nothing in your recent habits is working strongly against you right now."
        )

    ranked = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:2]

    if len(top) == 1:
        drivers_text = DRIVER_LABELS[top[0][0]]
    else:
        drivers_text = f"{DRIVER_LABELS[top[0][0]]} and {DRIVER_LABELS[top[1][0]]}"

    hint = IMPROVEMENT_HINTS[top[0][0]]

    return (
        f"Your risk appears {risk_tier.lower()} mainly due to {drivers_text}. "
        f"{hint} could meaningfully reduce your predicted risk."
    )


def _build_prompt(risk_score: float, risk_tier: str, persona: str, contributions: dict) -> str:
    breakdown_lines = "\n".join(f"- {k}: {v:+.2f}" for k, v in contributions.items())
    return (
        "You are a calm, factual wellness assistant narrating a risk-score breakdown "
        "that was already computed by a fixed formula (you are not scoring anything "
        "yourself). Only name a factor as a 'driver' if its contribution is notably "
        "large relative to the others - do not call small positive numbers 'elevated' "
        "just because they're the largest of a low set. If the tier is Low, or no factor "
        "clearly stands out, say plainly that things look fine right now instead of "
        "inventing a driver. Write exactly 2 short sentences: (1) name the 1-2 biggest "
        "genuine contributors in plain language, or note nothing stands out, (2) suggest "
        "one concrete, modest improvement (only if relevant) and its likely direction of "
        "effect. Do not diagnose, do not mention that you are an AI, and stay under 40 "
        "words total.\n\n"
        f"Risk score: {risk_score}/10 ({risk_tier})\n"
        f"Persona: {persona}\n"
        "Contribution breakdown (positive = increases risk, negative = protective):\n"
        f"{breakdown_lines}"
    )


def _llm_explanation_groq(prompt: str) -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # optional dependency - only imported if a key is set

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # free-tier friendly, low-latency
            max_tokens=120,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


def _llm_explanation_gemini(prompt: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # optional dependency - only imported if a key is set
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # free-tier model
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=120, temperature=0.4),
        )
        text = (response.text or "").strip()
        return text or None
    except Exception:
        return None


def _llm_explanation(risk_score: float, risk_tier: str, persona: str, contributions: dict) -> Optional[str]:
    """
    Tries Groq first (fast, generous free tier), then Gemini. Returns None
    if neither key is set or both calls fail - the caller falls back to the
    deterministic template either way.
    """
    prompt = _build_prompt(risk_score, risk_tier, persona, contributions)

    text = _llm_explanation_groq(prompt)
    if text:
        return text

    return _llm_explanation_gemini(prompt)


def explain(
    risk_score: float,
    risk_tier: str,
    persona: str,
    stress_level: float,
    sleep_quality: float,
    screen_time_hrs: float,
    happiness_index: float,
    exercise_frequency: float,
    days_without_social_media: float,
) -> dict:
    """Returns {"text": str, "source": "llm" | "rule_based"}."""
    contributions = decompose_risk(
        stress_level=stress_level,
        sleep_quality=sleep_quality,
        screen_time_hrs=screen_time_hrs,
        happiness_index=happiness_index,
        exercise_frequency=exercise_frequency,
        days_without_social_media=days_without_social_media,
    )

    llm_text = _llm_explanation(risk_score, risk_tier, persona, contributions)
    if llm_text:
        return {"text": llm_text, "source": "llm"}

    return {"text": _fallback_explanation(risk_score, risk_tier, contributions), "source": "rule_based"}
