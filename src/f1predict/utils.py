import json
from pathlib import Path


def ensure_dir(path):
    """Create a directory if it doesn't already exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    """Save a dictionary to a JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    """Load a JSON file into a dictionary"""
    with open(path) as f:
        return json.load(f)
