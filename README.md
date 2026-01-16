# F1 Race Pace & Pit Strategy Prediction (2025)

Predict:
1) **Next-lap pace** (regression: seconds)
2) **Pit window** (classification: probability of pitting within next K laps)

Data source: FastF1 (laps + weather + track status).

## Quickstart

### 1) Create venv + install
```bash
cd f1-race-pace-strategy

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
