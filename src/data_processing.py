"""
data_processing.py
-------------------
Cleans the raw digital-wellness survey data and engineers the features
PulseCheck AI's models are trained on:

    * Risk_Score        -> composite 0-10 burnout/stress-risk target
    * Persona_Cluster    -> raw KMeans cluster id (assigned in train_models.py)

Run directly to regenerate data/processed/wellness_features.csv:

    python src/data_processing.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "digital_wellness_raw.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "wellness_features.csv"

NUMERIC_COLS = [
    "Age",
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)",
    "Happiness_Index(1-10)",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise messy categorical text (stray spaces / capitalisation / typos)
    df["Social_Media_Platform"] = (
        df["Social_Media_Platform"]
        .astype(str)
        .str.strip()
        .replace(
            {
                "X ": "X",
                "Netfkix": "Netflix",
                "Youtube": "YouTube",
                "Message": "Other",
                "others": "Other",
                "Gmail": "Other",
                "Google Pay": "Other",
                "Paytm": "Other",
            }
        )
    )

    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()

    # Clip numeric survey scores into their valid 1-10 ranges (guards against
    # bad manual entry, e.g. a 1-10 field holding an 11)
    for col in ["Sleep_Quality(1-10)", "Stress_Level(1-10)", "Happiness_Index(1-10)"]:
        df[col] = df[col].clip(1, 10)

    df["Daily_Screen_Time(hrs)"] = df["Daily_Screen_Time(hrs)"].clip(0, 16)
    df["Exercise_Frequency(week)"] = df["Exercise_Frequency(week)"].clip(0, 14)
    df["Days_Without_Social_Media"] = df["Days_Without_Social_Media"].clip(0, 30)

    df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
    return df


def _minmax(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Scale a series to 0-10 given an assumed [lo, hi] domain, clipped."""
    return ((series.clip(lo, hi) - lo) / (hi - lo) * 10).clip(0, 10)


def decompose_risk(stress_level, sleep_quality, screen_time_hrs,
                    happiness_index, exercise_frequency,
                    days_without_social_media) -> dict:
    """
    Breaks compute_risk_score() into its per-factor weighted contributions.
    Positive values push risk up; negative values are protective. Used by
    the optional LLM explainer (and available for any future "why" UI)
    so the explanation always matches the score exactly - no separate
    approximation, no drift between the two.
    """
    screen_norm = min(10.0, max(0.0, screen_time_hrs / 10 * 10))
    sleep_deficit = 10 - sleep_quality
    happiness_deficit = 10 - happiness_index
    exercise_protective = min(10.0, max(0.0, exercise_frequency / 6 * 10))
    detox_protective = min(10.0, max(0.0, days_without_social_media / 7 * 10))

    return {
        "stress": round(0.32 * stress_level, 3),
        "sleep": round(0.22 * sleep_deficit, 3),
        "screen_time": round(0.16 * screen_norm, 3),
        "happiness": round(0.20 * happiness_deficit, 3),
        "exercise": round(-0.06 * exercise_protective, 3),
        "detox": round(-0.04 * detox_protective, 3),
    }


def compute_risk_score(stress_level, sleep_quality, screen_time_hrs,
                        happiness_index, exercise_frequency,
                        days_without_social_media) -> float:
    """
    The single source of truth for PulseCheck's 0-10 risk score.

    This is deliberately a transparent, auditable formula rather than a
    black-box model output. A RandomForestRegressor is also trained on this
    target (see train_models.py) to validate that it's learnable from raw
    inputs (R^2 ~ 0.97) and to power feature-importance analysis - but the
    score a person is actually shown, and the one the crisis-escalation
    threshold is checked against, comes from this function directly. On a
    dataset this small, a tree ensemble can under-predict combinations of
    extreme values it never saw in training; a safety-relevant escalation
    decision shouldn't depend on that kind of extrapolation gap.
    """
    screen_norm = min(10.0, max(0.0, screen_time_hrs / 10 * 10))
    sleep_deficit = 10 - sleep_quality
    happiness_deficit = 10 - happiness_index
    exercise_protective = min(10.0, max(0.0, exercise_frequency / 6 * 10))
    detox_protective = min(10.0, max(0.0, days_without_social_media / 7 * 10))

    raw_score = (
        0.32 * stress_level
        + 0.22 * sleep_deficit
        + 0.16 * screen_norm
        + 0.20 * happiness_deficit
        - 0.06 * exercise_protective
        - 0.04 * detox_protective
    )
    return round(min(10.0, max(0.0, raw_score)), 2)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the engineered columns PulseCheck's models consume."""
    df = df.copy()

    df["Risk_Score"] = df.apply(
        lambda r: compute_risk_score(
            stress_level=r["Stress_Level(1-10)"],
            sleep_quality=r["Sleep_Quality(1-10)"],
            screen_time_hrs=r["Daily_Screen_Time(hrs)"],
            happiness_index=r["Happiness_Index(1-10)"],
            exercise_frequency=r["Exercise_Frequency(week)"],
            days_without_social_media=r["Days_Without_Social_Media"],
        ),
        axis=1,
    )

    # Human-readable risk tier used across the UI / recommender
    df["Risk_Tier"] = pd.cut(
        df["Risk_Score"],
        bins=[-0.01, 3.5, 6.0, 8.5, 10.01],
        labels=["Low", "Moderate", "Elevated", "Critical"],
    )

    return df


def build_processed_dataset() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return df


if __name__ == "__main__":
    out = build_processed_dataset()
    print(f"Processed {len(out)} rows -> {PROCESSED_PATH}")
    print(out[["Risk_Score", "Risk_Tier"]].describe(include="all"))
