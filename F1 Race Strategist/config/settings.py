"""Central configuration. All paths resolve relative to the repo so the
project runs from any working directory."""

import os
from pathlib import Path

from dotenv import load_dotenv

ENGINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_ROOT.parent

load_dotenv(ENGINE_ROOT / ".env")

# FastF1 HTTP/session cache. The parent repo already holds 3.6 GB of
# cached 2022-2024 race data, so we reuse it instead of re-downloading.
CACHE_DIR = Path(os.getenv("F1_CACHE_DIR", REPO_ROOT / "f1_cache"))

# Processed datasets and model artifacts live inside the engine.
# Env overrides exist so builds can target scratch storage in CI or
# sandboxed environments.
STORE_DIR = Path(os.getenv("F1_STORE_DIR", ENGINE_ROOT / "data" / "store"))
ARTIFACTS_DIR = Path(os.getenv("F1_ARTIFACTS_DIR", ENGINE_ROOT / "models" / "artifacts"))
STORE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Seasons with cached race data used for model training.
TRAIN_SEASONS = [2022, 2023, 2024]

# Live tracker season. FastF1 resolves the current schedule at runtime.
CURRENT_SEASON = 2026

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Track status codes used by FastF1 / F1 live timing.
# 1 all clear, 2 yellow, 4 safety car, 5 red flag, 6 VSC, 7 VSC ending.
SC_CODE = "4"
VSC_CODES = ("6", "7")
RED_FLAG_CODE = "5"
