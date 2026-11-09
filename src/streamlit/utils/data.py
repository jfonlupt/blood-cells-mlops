"""Chargement des données partagées entre les pages de la présentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "report"
LEADERBOARD_PATH = PROJECT_ROOT / "results" / "leaderboard.csv"


def figure(name: str) -> str:
    """Chemin absolu vers une figure du rapport, pour ``st.image``."""
    return str(FIGURES_DIR / name)


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    """Charge le journal cumulatif des runs d'entraînement (``results/leaderboard.csv``)."""
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LEADERBOARD_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
