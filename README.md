# F1 Race Strategy Predictor

A machine learning project that predicts **lap times** and **pit stop windows** for Formula 1 races, built using the [FastF1](https://docs.fastf1.dev/) Python library and scikit-learn / XGBoost.

This was my final year project. The idea came from watching F1 Strategy Group decisions and wondering if you could predict when a driver *should* pit just from timing data.

---

## What does it do?

There are two prediction tasks:

1. **Lap time prediction (regression)** — given everything we know at the start of a lap (previous lap time, sector splits, tire age, weather, track status), predict how long the next lap will take in seconds.

2. **Pit stop window prediction (classification)** — predict whether a driver is likely to pit within the next N laps (default: 3). This is a binary label: 1 = pit coming soon, 0 = not yet.

The models are trained on historical race data fetched directly from the FastF1 API (lap timing, weather, and track status). After training, there's a Streamlit web app where you can explore predictions race-by-race and even edit scenario parameters to see how they affect the output.

---

## Project Structure

```text
F1-Race-Strategy/
│
├── app/
│   └── streamlit_app.py        # Interactive web dashboard
│
├── src/f1predict/
│   ├── config.py               # Path settings and defaults
│   ├── data.py                 # Downloads and cleans FastF1 data
│   ├── features.py             # Feature engineering + label creation
│   ├── modeling.py             # Model building, preprocessing, evaluation
│   ├── train.py                # Main training script (run this first)
│   └── utils.py                # Small helper functions
│
├── tests/
│   ├── test_features.py        # Tests for pit label logic
│   └── test_modeling.py        # Tests for pipeline fit/predict
│
├── data/                       # Raw + processed data saved here (gitignored)
├── models/                     # Trained model files saved here (gitignored)
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### Requirements

- Python 3.10+
- Internet connection (for first-time data download from FastF1)

### Install

```bash
# Clone the repo
git clone https://github.com/DhruvB100/F1-Race-Strategy.git
cd F1-Race-Strategy

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the project package in editable mode
pip install -e .
```

XGBoost is listed in `requirements.txt` as an optional but recommended dependency. The code will fall back to scikit-learn's `GradientBoosting` if it's not installed.

---

## How to Use

### Step 1 — Train the models

This will download race data for the selected year, engineer features, run cross-validation, and save the trained model.

```bash
python -m f1predict.train --year 2024
```

Additional options:

```bash
# Only load first 5 races (good for a quick test run)
python -m f1predict.train --year 2024 --max-events 5

# Change how far ahead we predict pit stops (default is 3 laps)
python -m f1predict.train --year 2024 --pit-horizon 5

# Don't use XGBoost even if it's installed
python -m f1predict.train --year 2024 --no-xgb
```

Training will print progress as it loads each event, then show cross-validation scores at the end. The model is saved to `models/model_{year}.joblib`.

> **Note:** The first run will be slow because FastF1 downloads race data from the internet. Subsequent runs use the local cache in `data/fastf1_cache/`.

### Step 2 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

This opens a browser window. Use the sidebar to:

- Select a season year, Grand Prix, and driver
- Choose a specific lap to inspect
- Edit scenario inputs (tire compound, track temp, tire age, previous lap time) to see how predictions change in real time

---

## How the Machine Learning Works

### Features (model inputs)

All features are based on information available at the **start** of a lap — no future data is used (no leakage).

| Feature | Description |
| --- | --- |
| `prev_lap_time_s` | Previous lap time in seconds |
| `prev_Sector1/2/3_s` | Previous sector times |
| `prev_gap_to_ahead_s` | Gap to the car ahead last lap |
| `stint_lap` | How many laps this tire set has done (tire age) |
| `Stint` | Which stint number (1st set, 2nd set, etc.) |
| `Compound` | Tire type: SOFT, MEDIUM, HARD |
| `TrackTemp` | Track surface temperature |
| `AirTemp` | Ambient air temperature |
| `Humidity` | Humidity % |
| `Rainfall` | Whether it was raining |
| `track_status` | Green, yellow flag, safety car, etc. |
| `Position` | Race position at that lap |
| `Driver`, `Team` | Driver and constructor (one-hot encoded) |
| `event_name` | Which Grand Prix (one-hot encoded) |

### Preprocessing

The preprocessing pipeline handles:

- **Missing values**: filled with median (numeric) or most frequent value (categorical)
- **Categorical encoding**: one-hot encoding via `OneHotEncoder(handle_unknown="ignore")` so unseen categories don't crash the model

### Models

- **Regression** (lap time): `XGBRegressor` with 300 trees, learning rate 0.1, depth 5
- **Classification** (pit window): `XGBClassifier` with 300 trees, learning rate 0.1, depth 4
- Fallback to `GradientBoostingRegressor` / `GradientBoostingClassifier` if XGBoost isn't installed

### Cross-validation

We use `GroupKFold` with race events as groups — meaning the model is never trained and tested on laps from the same Grand Prix. This is important because laps within a race are highly correlated, and mixing them would give artificially good scores.

Evaluation metrics:

- Regression: **Mean Absolute Error** (in seconds)
- Classification: **F1 score** (accounts for class imbalance — pit laps are rare)

### Labels

- `lap_time_s`: the actual lap time in seconds (regression target)
- `pit_within_k`: 1 if the driver pits in the next `k` laps, 0 otherwise (classification target)

Pit detection works by checking:

1. Did the stint number increase on the next lap? (driver changed tires)
2. Is `PitInTime` or `PitOutTime` recorded for this lap?

---

## Running the Tests

```bash
pytest -v
```

There are two test files:

- `test_features.py` — checks the pit window labeling logic with a synthetic 6-lap race
- `test_modeling.py` — smoke test that pipelines can fit and predict without crashing

---

## Data Source

All race data comes from [FastF1](https://docs.fastf1.dev/), which is an unofficial Python client for the F1 timing API. It provides:

- Lap-by-lap timing data (lap time, sector splits, tire info)
- Weather data (track temp, air temp, humidity, wind, rain)
- Track status data (safety car periods, yellow flags, etc.)

Data is cached locally after the first download, so you only need the internet connection once per season.

---

## Known Limitations

- **Qualifying and sprint sessions** are not included (only race data by default)
- **Pit stop predictions are harder to get right** because pit stops are rare events (~2-3 per driver per race), so the classifier has to deal with significant class imbalance
- The model doesn't use car-specific telemetry (throttle, brakes, engine modes) — that would likely improve lap time prediction but FastF1 telemetry is slow to download
- Predictions are only as good as the training data — a model trained on 2024 data might not generalise perfectly to 2025 if the regulations changed significantly

---

## Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `fastf1` | ≥3.4.0 | F1 data API |
| `pandas` | ≥2.2.0 | Data manipulation |
| `numpy` | ≥1.26.0 | Numerical operations |
| `scikit-learn` | ≥1.4.0 | ML pipelines and models |
| `xgboost` | ≥2.0.0 | Gradient boosting (optional) |
| `streamlit` | ≥1.31.0 | Web dashboard |
| `matplotlib` | ≥3.8.0 | Charts |
| `joblib` | ≥1.3.0 | Saving/loading models |
| `pyarrow` | ≥15.0.0 | Parquet file support |
| `pytest` | ≥8.0.0 | Running tests |
