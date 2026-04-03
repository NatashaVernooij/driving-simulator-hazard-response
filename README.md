# Driving Simulator Hazard Response Analysis

Analysis pipeline for a driving simulator study examining how lighting and roadway cue conditions affect driver responses to a looming hazard vehicle.

## Study Overview

Participants drove a simulated highway and encountered a stationary hazard vehicle. The script processes raw simulator output to:

- Clean and align multi-participant CSV data
- Calculate vehicle distances (bumper-to-bumper and to camera)
- Retroactively verify hazard collision detection
- Extract the first **throttle deceleration**, **brake engagement**, and **steering change** per trial
- Compute the traditional **Psycho-Looming Distance (PLD)** and compare it to actual response distances
- Run mixed-effects regression models (`DistanceToHazard ~ Condition + Speed | Participant`)

**Conditions:** DayFull (day + road cues), DayNoCues (day, no cues), NightNoCues (night, no cues)

---

## Repository Structure

```
simulator-hazard-response/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── example/          ← three anonymised example participant CSVs
│       ├── 001.csv
│       ├── 002.csv
│       └── 003.csv
│
├── output/               ← created automatically on first run (gitignored)
│
└── analysis/
    └── hazard_analysis.py
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/simulator-hazard-response.git
cd simulator-hazard-response
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python analysis/hazard_analysis.py
```

All output files (PDFs, PNGs, CSVs) are written to `output/`. The folder is created automatically if it does not exist.

---

## Input Data Format

Each CSV corresponds to one participant, named by participant ID (e.g. `001.csv`). The script expects the following columns:

| Column | Description |
|--------|-------------|
| `TrialNumber` | Trial identifier |
| `LightCondition` | `Day` or `Night` |
| `CuesCondition` | `Full` or `NoCues` |
| `TurnCondition` | `NoTurn` or `Turn` |
| `HazardCondition(Spawn)` | Boolean — was a hazard spawned? |
| `Time` | Absolute timestamp |
| `TrialTime` | Time since trial start (s) |
| `SVPositionX/Z` | Subject vehicle (player) position |
| `LVPositionX/Z` | Lead vehicle (hazard) position |
| `SpeedSV` | Player speed (mph) |
| `SpeedLV` | Hazard speed (mph) |
| `Throttle%` | Throttle pedal position (0–100) |
| `SteerAngle` | Steering wheel angle (degrees) |
| `BrakeSwitch` | Brake pedal contact (0/1) |
| `BrakePedal%` | Brake pedal pressure (0–100) |
| `IsCollided` | Collision flag from Unity |
| `HazardCollision` | Hazard-specific collision flag |

---

## Key Outputs

| File | Description |
|------|-------------|
| `all_hazard_data.csv` | Cleaned full dataset (all participants, all hazard trials) |
| `hazard_data.csv` | After exclusions |
| `trial_responses.csv` | First response per trial — main analysis file |
| `all_collisions.csv` | All collision events (for manual verification) |
| `distance_by_time.pdf` | Z-position over time, one page per trial |
| `player_behaviour_per_trial_relative_to_hazard.pdf` | Per-trial 3-panel behaviour plots |
| `mean_throttle_by_distance.png` | LOWESS grand-mean throttle curve |
| `mean_brake_by_distance.png` | LOWESS grand-mean brake curve |
| `mean_steerangle_by_distance.png` | LOWESS grand-mean steer-angle curve |
| `brake_engagement_histogram.png` | Distribution of brake onset distances |
| `response_check_*.png` | Randomly sampled trial response visualisations |
| `dist_to_hazard_by_response_type.png` | Bar chart — response distance by type and condition |

---

## Configuration

All key parameters are defined near the top of `hazard_analysis.py`:

```python
DATA_DIR      = BASE_DIR / "data" / "example"   # change if data is elsewhere
ROLL_WINDOW   = 10      # smoothing window (samples)
SUSTAINED_SEC = 0.1     # min duration to count as a valid response
THROTTLE_DROP_STD = 3   # std-dev multiplier for throttle threshold
STEER_STD_MULT    = 3   # std-dev multiplier for steer threshold
STEER_THRESHOLD   = 5   # minimum steer change (degrees)
```

---

## Citation

If you use this code, please cite:

> [Your name(s), year, title, journal/conference — update before publishing]

---

## License

MIT License — see `LICENSE` for details.
