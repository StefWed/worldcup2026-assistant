"""
Shared configuration: API client setup and constants.
"""

import os
import warnings
from openai import OpenAI

# Silence the known langchain-community deprecation warning (PyPDFLoader).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

# --- OpenAI client ---
# Reads OPENAI_API_KEY from the environment. When running via `streamlit run`,
# you can also put it in .streamlit/secrets.toml as OPENAI_API_KEY = "sk-..."
# and set it into the environment below (see app.py for the secrets fallback).
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
AGENT_MODEL = "gpt-3.5-turbo"

# --- Team stats for the analyze_match tool ---
TEAM_STATS = {
    "France":      {"win_rate": 0.75, "avg_goals": 2.0, "goals_conceded": 0.8},
    "Spain":       {"win_rate": 0.72, "avg_goals": 1.9, "goals_conceded": 0.6},
    "Argentina":   {"win_rate": 0.70, "avg_goals": 1.8, "goals_conceded": 0.9},
    "England":     {"win_rate": 0.65, "avg_goals": 1.6, "goals_conceded": 1.0},
    "Switzerland": {"win_rate": 0.58, "avg_goals": 1.3, "goals_conceded": 0.9},
    "Norway":      {"win_rate": 0.55, "avg_goals": 1.7, "goals_conceded": 1.2},
}
