from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    cache_dir: Path = PROJECT_ROOT / "data" / "fastf1_cache"

PATHS = Paths()

DEFAULT_PIT_HORIZON_LAPS = 3  # classify "pit within next K laps"
DEFAULT_SESSION = "R"         # Race
