from pathlib import Path

# Get the root directory of the project
# This file is at: PROJECT_ROOT/src/f1predict/config.py
# So we need to go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where we store data and trained models
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "fastf1_cache"  # fastf1 caches downloaded data here

# Default prediction settings
PIT_HORIZON = 3       # predict if driver will pit within next 3 laps
DEFAULT_SESSION = "R"  # R = Race (as opposed to Q = Qualifying)
