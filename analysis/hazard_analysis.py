#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driving Simulator Hazard Response Analysis
===========================================
Loads per-participant CSV files from a driving simulator, cleans and processes
the data, computes response metrics (throttle deceleration, brake engagement,
steering change), and runs mixed-effects regression models.

Usage
-----
1. Place participant CSV files in data/example/
2. Run:  python analysis/hazard_analysis.py
3. Outputs (PDFs, CSVs, PNGs) are written to output/

Requirements
------------
See requirements.txt
"""

# %% Packages
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.formula.api as smf
import pingouin as pg
from pathlib import Path

# %% Paths — edit DATA_DIR if your CSVs live somewhere else
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
DATA_DIR = BASE_DIR / "data" / "example"            # input CSVs
OUT_DIR  = BASE_DIR / "output"                      # all outputs go here

# Output subfolders
FIGURES_DIR = OUT_DIR / "figures"       # all plots and PDFs

for d in [FIGURES_DIR,
          FIGURES_DIR / "trial_plots",
          FIGURES_DIR / "response_curves",
          FIGURES_DIR / "summary",
          FIGURES_DIR / "response_checks"]:
    d.mkdir(parents=True, exist_ok=True)

# %% Vehicle dimensions (metres)
HAZARD_WIDTH   = 1.67
HAZARD_LENGTH  = 4.09
PLAYER_WIDTH   = 1.84
PLAYER_LENGTH  = 4.97

hazard_half_length = HAZARD_LENGTH / 2
hazard_half_width  = HAZARD_WIDTH  / 2
player_half_length = PLAYER_LENGTH / 2
player_half_width  = PLAYER_WIDTH  / 2

CENTER_DIST_TO_CAMERA = 0.15   # distance from player camera to car centre

# %% Detection / response parameters
ROLL_WINDOW          = 10    # smoothing window (samples)
SUSTAINED_SEC        = 0.1   # min duration above threshold to count as response
THROTTLE_DROP_STD    = 3     # std-dev multiplier for adaptive throttle threshold
STEER_STD_MULT       = 3     # std-dev multiplier for adaptive steer threshold
STEER_THRESHOLD      = 5     # minimum absolute steer-angle change (degrees)
LOCAL_WINDOW_SEC     = 0.5   # seconds after max-throttle used to estimate variability

# %% Participants whose HazardCollision flag from Unity is reliable
HAZARD_COLLIDED_WORKS = [
    '032', '039', '087', '015', '052', '065', '072', '099',
    '023', '058', '045', '070', '009', '006', '013', '063', '028'
]

# Participant / trial pairs to exclude from analysis
EXCLUDE_TRIALS = [
    ('003', 1), ('007', 1), ('012', 1), ('018', 18),
    ('028', 16), ('028', 18), ('035', 12), ('056', 1), ('061', 9)
]
EXCLUDE_PARTICIPANTS = ['085']


# ============================================================
# 1. DATA LOADING AND CLEANING
# ============================================================

data_files = glob.glob(str(DATA_DIR / "*.csv"))
print(f"Found {len(data_files)} CSV file(s) in {DATA_DIR}")

hazard_data = pd.DataFrame()

for ifile in data_files:
    print(f"  Loading {os.path.basename(ifile)}")
    df = pd.read_csv(ifile, skipinitialspace=True)
    basename = os.path.splitext(os.path.basename(ifile))[0]
    df['ParticipantID'] = basename

    # Fill missing condition labels
    df['LightCondition'] = df['LightCondition'].fillna('Day')
    df['CuesCondition']  = df['CuesCondition'].fillna('NoCues')
    df['TurnCondition']  = df['TurnCondition'].fillna('NoTurn')
    df['IsCollided']     = df['IsCollided'].fillna('No')

    # Keep only straight-road hazard trials
    hazard_trials = df[
        (df['HazardCondition(Spawn)'] == True) &
        (df['TurnCondition'] == 'NoTurn')
    ]
    hazard_data = pd.concat([hazard_data, hazard_trials], ignore_index=True)

# Move ParticipantID to front and rename columns
pop_col = hazard_data.pop('ParticipantID')
hazard_data.insert(0, 'ParticipantID', pop_col)
hazard_data.rename(columns={
    'HazardCondition(Spawn)': 'HazardCondition',
    'SpeedSV':    'PlayerSpeed',
    'SpeedLV':    'HazardSpeed',
    'Throttle%':  'ThrottlePercent',
    'BrakePedal%': 'BrakePercent'
}, inplace=True)

# Participants 001 and 002 had a brake sensor that reported double values
hazard_data.loc[
    hazard_data['ParticipantID'].isin(['001', '002']),
    'BrakePercent'
] /= 2

hazard_data['HazardCollision'] = hazard_data['HazardCollision'].fillna(False)

# --- Compute vehicle centre positions ---
hazard_data['PlayerPositionX'] = hazard_data['SVPositionX'] - 0.04
hazard_data['PlayerPositionZ'] = hazard_data['SVPositionZ'] - 2.36
hazard_data['HazardPositionX'] = hazard_data['LVPositionX']
hazard_data['HazardPositionZ'] = hazard_data['LVPositionZ'] - 2.17

# Centre-to-centre and bumper-to-bumper distances
hazard_data['DistanceBetweenCarCenters'] = (
    abs(hazard_data['PlayerPositionZ']) - abs(hazard_data['HazardPositionZ'])
).round(3)

hazard_data['DistanceBumperToBumper'] = np.where(
    abs(hazard_data['DistanceBetweenCarCenters']) <= (hazard_half_length + player_half_length),
    0,
    np.where(
        hazard_data['DistanceBetweenCarCenters'] < 0,
        hazard_data['DistanceBetweenCarCenters'] + (hazard_half_length + player_half_length),
        hazard_data['DistanceBetweenCarCenters'] - (hazard_half_length + player_half_length)
    )
).round(3)

hazard_data['PlayerDistanceToHazard'] = (
    hazard_data['DistanceBumperToBumper'] + player_half_length - CENTER_DIST_TO_CAMERA
)

# --- Recalculate HazardCollision for older participants ---
overlap_z = (
    np.abs(hazard_data['LVPositionZ'] - hazard_data['SVPositionZ']) <=
    (hazard_half_length + player_half_length)
)
overlap_x = (
    np.abs(hazard_data['HazardPositionX'] - hazard_data['PlayerPositionX']) <=
    (hazard_half_width + player_half_width)
)
speed_drop = (
    hazard_data.groupby(['ParticipantID', 'TrialNumber'])['PlayerSpeed'].diff() <= -2
)
collision_occurs = overlap_x & overlap_z & speed_drop

hazard_data['HazardCollision'] = np.where(
    hazard_data['ParticipantID'].isin(HAZARD_COLLIDED_WORKS),
    hazard_data['HazardCollision'],
    collision_occurs
)

# --- Save collision list for manual cross-reference ---
all_collisions = hazard_data[hazard_data['HazardCollision'] == True][[
    'ParticipantID', 'TrialNumber', 'TrialTime',
    'HazardCollision', 'ThrottlePercent', 'BrakePercent', 'SteerAngle'
]]
all_collisions.to_csv(OUT_DIR / 'all_collisions.csv', index=False)

# --- Condition label and trial order ---
hazard_data['Condition'] = hazard_data['LightCondition'] + hazard_data['CuesCondition']

unique_trials = (
    hazard_data[['ParticipantID', 'Condition', 'TrialNumber']]
    .drop_duplicates()
    .sort_values(['ParticipantID', 'Condition', 'TrialNumber'])
    .reset_index(drop=True)
)
unique_trials['ConditionTrialOrder'] = (
    unique_trials.groupby(['ParticipantID', 'Condition']).cumcount() + 1
)
hazard_data = hazard_data.merge(
    unique_trials, on=['ParticipantID', 'Condition', 'TrialNumber'], how='left'
)

col_names = [
    'ParticipantID', 'TrialNumber', 'LightCondition', 'CuesCondition',
    'Condition', 'ConditionTrialOrder', 'TurnCondition', 'HazardCondition',
    'Time', 'TrialTime', 'PlayerPositionX', 'PlayerPositionZ',
    'HazardPositionX', 'HazardPositionZ', 'PlayerSpeed', 'HazardSpeed',
    'ThrottlePercent', 'SteerAngle', 'BrakeSwitch', 'BrakePercent',
    'IsCollided', 'HazardCollision', 'DistanceBetweenCarCenters',
    'DistanceBumperToBumper', 'PlayerDistanceToHazard'
]
hazard_data = hazard_data[col_names]
hazard_data.to_csv(OUT_DIR / 'all_hazard_data.csv', index=False)


# ============================================================
# 2. PRELIMINARY PLOTS: Z position by time (one page per trial)
# ============================================================

print("Generating distance-by-time PDF…")
with PdfPages(FIGURES_DIR / "trial_plots" / "distance_by_time.pdf") as pdf:
    for participant, p_group in hazard_data.groupby('ParticipantID'):
        for trial, t_group in p_group.groupby('TrialNumber'):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(t_group['TrialTime'], t_group['PlayerPositionZ'],
                    marker='o', label="Path")
            hazard_pos = t_group['HazardPositionZ'].iloc[0]
            ax.axhline(y=hazard_pos, color='red', linestyle='--', linewidth=2)
            ax.text(
                x=t_group['TrialTime'].min(), y=hazard_pos + 0.5,
                s=f"Hazard Position = {hazard_pos}",
                color='red', va='bottom', ha='left', fontsize=10
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-2200, -1350)
            ax.set_title(f'Participant {participant} — Trial {trial}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Z Position')
            ax.grid(True)
            pdf.savefig(fig)
            plt.close(fig)

# ============================================================
# 3. PLAYER BEHAVIOUR RELATIVE TO HAZARD (one page per trial)
# ============================================================

print("Generating per-trial behaviour PDF…")
with PdfPages(FIGURES_DIR / "trial_plots" / "player_behaviour_per_trial_relative_to_hazard.pdf") as pdf:
    for participant, p_group in hazard_data.groupby('ParticipantID'):
        for trial, t_group in p_group.groupby('TrialNumber'):
            fig = plt.figure(figsize=(10, 8), constrained_layout=True)
            gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1])

            # Left: X vs distance-to-hazard
            ax1 = fig.add_subplot(gs[:, 0])
            ax1.plot(t_group['PlayerPositionX'], t_group['DistanceBumperToBumper'],
                     marker='o', markersize=2, label="Path")

            hazardX = t_group['HazardPositionX'].iloc[0]
            hazardZ = 0 - hazard_half_length
            ax1.plot(hazardX, hazardZ, marker='*', color='black', markersize=3)
            hazardLeft   = hazardX - hazard_half_width
            hazardBottom = -HAZARD_LENGTH
            ax1.add_patch(Rectangle(
                (hazardLeft, hazardBottom), HAZARD_WIDTH, HAZARD_LENGTH,
                edgecolor='red', facecolor='red', linewidth=2, alpha=0.8
            ))

            last_x = t_group['PlayerPositionX'].iloc[-1]
            last_y = t_group['DistanceBumperToBumper'].iloc[-1]
            ax1.add_patch(Rectangle(
                (last_x - player_half_width, last_y), PLAYER_WIDTH, PLAYER_LENGTH,
                edgecolor='grey', facecolor='grey', linewidth=2
            ))

            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-50, 700)
            ax1.invert_yaxis()
            ax1.set_title(f'Participant {participant} — Trial {trial}')
            ax1.set_xlabel('Player X Position (m)')
            ax1.set_ylabel('Player Distance to Hazard (m)')
            ax1.grid(True)

            # Top-right: speed / throttle / brake vs distance
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(t_group['ThrottlePercent'], t_group['DistanceBumperToBumper'],
                     color='green', marker='o', markersize=2, alpha=0.5, label='Throttle %')
            ax2.plot(t_group['PlayerSpeed'], t_group['DistanceBumperToBumper'],
                     color='blue', marker='d', markersize=2, alpha=0.5, label='Speed (mph)')
            ax2.plot(t_group['BrakePercent'], t_group['DistanceBumperToBumper'],
                     color='goldenrod', marker='x', markersize=2, label='Brake %')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlim(0, 100)
            ax2.set_ylim(-50, 700)
            ax2.invert_yaxis()
            ax2.set_xlabel('Percent / MPH')
            ax2.legend()
            ax2.grid(True)

            # Bottom-right: steer angle vs distance
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(t_group['SteerAngle'], t_group['DistanceBumperToBumper'],
                     color='blue', marker='o', markersize=2)
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax3.set_xlim(-540, 540)
            ax3.set_ylim(-50, 700)
            ax3.invert_yaxis()
            ax3.set_xlabel('Steer Angle (degrees)')
            ax3.grid(True)

            pdf.savefig(fig)
            plt.close(fig)


# ============================================================
# 4. EXCLUSIONS
# ============================================================

mask = (
    hazard_data[['ParticipantID', 'TrialNumber']]
    .apply(tuple, axis=1)
    .isin(EXCLUDE_TRIALS)
)
mask = mask | hazard_data['ParticipantID'].isin(EXCLUDE_PARTICIPANTS)
hazard_data = hazard_data[~mask]
hazard_data.to_csv(OUT_DIR / 'hazard_data.csv', index=False)


# ============================================================
# 5. BRAKING SUMMARY STATS AND PLOTS
# ============================================================

grouped = hazard_data.groupby(['ParticipantID', 'TrialNumber'])
trial_summary = grouped['BrakeSwitch'].apply(lambda x: (x == 1).any()).reset_index(name='BrakeUsed')

total_trials        = len(trial_summary)
trials_with_brake   = trial_summary['BrakeUsed'].sum()
pct_with_brake      = trials_with_brake / total_trials * 100

print(f"\nBraking summary")
print(f"  Total trials:               {total_trials}")
print(f"  Trials with brake pressed:  {trials_with_brake}")
print(f"  Percent with braking:       {pct_with_brake:.1f}%")

brake_depressed = hazard_data[
    (hazard_data['BrakeSwitch'] == 1) &
    (hazard_data['PlayerDistanceToHazard'] <= 500)
]

print(f"\n  Mean bumper-to-bumper distance when braking: "
      f"{brake_depressed['DistanceBumperToBumper'].mean():.2f} m "
      f"(SD={brake_depressed['DistanceBumperToBumper'].std():.2f})")

# Histogram: brake engagement distance
fig, ax = plt.subplots()
ax.hist(brake_depressed['DistanceBumperToBumper'], bins=20, edgecolor='black', alpha=0.7)
ax.invert_xaxis()
ax.set_xlabel("Bumper-to-Bumper Distance from Hazard (m)")
ax.set_ylabel("Frequency")
ax.set_title("Braking Occurrences by Distance from Hazard")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "response_curves" / 'brake_engagement_histogram.png', dpi=150)
plt.close(fig)

print("  Saved brake_engagement_histogram.png")


# ============================================================
# 6. SMOOTHED GRAND-MEAN RESPONSE CURVES BY DISTANCE
# ============================================================

bin_width = 1
bins = range(-50, 701, bin_width)
hazard_data['distance_bin'] = pd.cut(hazard_data['DistanceBumperToBumper'], bins=bins)

def plot_response_by_distance(measure_col, ylabel, title, out_name,
                               use_abs=False, xlim=(-51, 201),
                               subfolder="response_curves"):
    """Plot per-participant LOWESS curves + grand-mean LOWESS curve."""
    if use_abs:
        summary = (
            hazard_data.groupby(['ParticipantID', 'distance_bin'], observed=True)[measure_col]
            .apply(lambda x: x.abs().mean())
            .reset_index()
        )
    else:
        summary = (
            hazard_data.groupby(['ParticipantID', 'distance_bin'], observed=True)[measure_col]
            .mean()
            .reset_index()
        )
    summary['bin_center'] = summary['distance_bin'].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(12, 7))
    for pid in summary['ParticipantID'].unique():
        part_df = summary[summary['ParticipantID'] == pid].sort_values('bin_center')
        if len(part_df) > 10:
            sm = lowess(part_df[measure_col], part_df['bin_center'], frac=0.05)
            ax.plot(sm[:, 0], sm[:, 1], color='gray', alpha=0.2, linewidth=1)

    grand = (
        summary.groupby('bin_center')[measure_col]
        .mean().reset_index().sort_values('bin_center')
    )
    grand_sm = lowess(grand[measure_col], grand['bin_center'], frac=0.05)
    ax.plot(grand_sm[:, 0], grand_sm[:, 1], color='red', linewidth=3, label='Grand Mean')
    ax.plot([], [], color='gray', alpha=0.4, linewidth=1, label='Participant Means')

    ax.set_xlabel('Bumper-to-Bumper Distance (m)')
    ax.set_xlim(*xlim)
    ax.invert_xaxis()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / subfolder / out_name, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_name}")

print("\nGenerating response-by-distance plots…")
plot_response_by_distance('ThrottlePercent', 'Mean Throttle (%)',
    'Throttle Percent vs Bumper-to-Bumper Distance to Hazard',
    'mean_throttle_by_distance.png', xlim=(-51, 701))

plot_response_by_distance('BrakePercent', 'Mean Brake (%)',
    'Brake Percent vs Distance to Hazard',
    'mean_brake_by_distance.png', xlim=(-51, 201))

plot_response_by_distance('SteerAngle', 'Mean Steer Angle (absolute value)',
    'Absolute Steer Angle vs Distance to Hazard',
    'mean_steerangle_by_distance.png', use_abs=True, xlim=(-51, 126))


# ============================================================
# 7. PLD AND FIRST RESPONSE EXTRACTION
# ============================================================

# Compute traditional psycho-looming distance
hazard_data['TradPLD'] = (
    np.sqrt(
        (hazard_data['PlayerSpeed'] * (5280 / 3600) * HAZARD_WIDTH * 3.28084) / 0.006
    ) * 0.3048
)

results = []

for (pid, trial), df in hazard_data.groupby(['ParticipantID', 'TrialNumber']):
    df = df.sort_values('TrialTime').copy()

    # --- PLD intersection ---
    diff = df['TradPLD'] - df['PlayerDistanceToHazard']
    sign_change = diff.shift(1) * diff < 0

    if diff.eq(0).any():
        idx_eq = diff[diff == 0].index[0]
        pld_val   = df.loc[idx_eq, 'PlayerDistanceToHazard']
        pld_time  = df.loc[idx_eq, 'TrialTime']
        pld_speed = df.loc[idx_eq, 'PlayerSpeed']
    elif sign_change.any():
        idx2 = sign_change[sign_change].index[0]
        idx1 = df.index[df.index.get_loc(idx2) - 1]
        x1, x2 = df.loc[idx1, 'TrialTime'], df.loc[idx2, 'TrialTime']
        y1d, y2d = diff.loc[idx1], diff.loc[idx2]
        t = abs(y1d) / (abs(y1d) + abs(y2d))
        pld_time  = x1 + t * (x2 - x1)
        pld_val   = df.loc[idx1, 'PlayerDistanceToHazard'] + t * (
            df.loc[idx2, 'PlayerDistanceToHazard'] - df.loc[idx1, 'PlayerDistanceToHazard']
        )
        pld_speed = df.loc[idx1, 'PlayerSpeed'] + t * (
            df.loc[idx2, 'PlayerSpeed'] - df.loc[idx1, 'PlayerSpeed']
        )
    else:
        pld_val = pld_time = pld_speed = np.nan

    # --- Hazard appearance time ---
    hazard_distance = 500 - player_half_length + CENTER_DIST_TO_CAMERA
    hazard_rows = df[df['DistanceBumperToBumper'] <= hazard_distance]
    if hazard_rows.empty:
        continue
    hazard_in_view_time = hazard_rows['TrialTime'].iloc[0]

    # --- Throttle baseline and adaptive threshold ---
    df['ThrottleSmooth'] = df['ThrottlePercent'].rolling(ROLL_WINDOW, min_periods=1).mean()
    post_hazard_df = df[
        (df['TrialTime'] >= hazard_in_view_time) &
        (df['DistanceBumperToBumper'] >= 100)
    ]
    if post_hazard_df.empty:
        continue
    max_throttle_idx   = post_hazard_df['ThrottleSmooth'].idxmax()
    max_throttle_value = df.loc[max_throttle_idx, 'ThrottleSmooth']
    max_throttle_time  = df.loc[max_throttle_idx, 'TrialTime']

    local_window = df[
        (df['TrialTime'] >= max_throttle_time) &
        (df['TrialTime'] <= max_throttle_time + LOCAL_WINDOW_SEC)
    ]
    throttle_std       = local_window['ThrottleSmooth'].std()
    throttle_threshold = max_throttle_value - (THROTTLE_DROP_STD * throttle_std)

    time_diffs       = df['TrialTime'].diff().median()
    sustained_samples = max(1, int(SUSTAINED_SEC / time_diffs))

    # --- Steer baseline and adaptive threshold ---
    df['SteerSmooth'] = df['SteerAngle'].rolling(ROLL_WINDOW, min_periods=1).mean()
    steer_df = df[(df['DistanceBumperToBumper'] >= 115) & (df['DistanceBumperToBumper'] <= 400)]
    if steer_df.empty:
        continue
    steer_mean             = steer_df['SteerSmooth'].mean()
    steer_std              = steer_df['SteerSmooth'].std()
    steer_threshold_lower  = steer_mean - (STEER_STD_MULT * steer_std)
    steer_threshold_upper  = steer_mean + (STEER_STD_MULT * steer_std)

    # --- Detect response onsets ---
    post_hazard    = df[df['TrialTime'] >= max_throttle_time]
    throttle_mask  = post_hazard['ThrottleSmooth'] < throttle_threshold
    throttle_sus   = throttle_mask.rolling(sustained_samples, min_periods=1).sum() >= sustained_samples
    throttle_idx   = throttle_sus[throttle_sus].index.min()

    brake_idx = df.index[
        (df['TrialTime'] >= hazard_in_view_time) & (df['BrakeSwitch'] == 1)
    ].min()

    steer_det  = df[df['DistanceBumperToBumper'] <= 115]
    steer_mask = (
        ((steer_det['SteerSmooth'] < steer_threshold_lower) &
         (steer_det['SteerSmooth'] <= steer_mean - STEER_THRESHOLD)) |
        ((steer_det['SteerSmooth'] > steer_threshold_upper) &
         (steer_det['SteerSmooth'] >= steer_mean + STEER_THRESHOLD))
    )
    steer_sus  = steer_mask.rolling(sustained_samples, min_periods=1).sum() >= sustained_samples
    steer_idx  = steer_sus[steer_sus].index.min()

    temp_rows = []
    for idx, label, col in zip(
        [throttle_idx, brake_idx, steer_idx],
        ['ThrottleDecel', 'BrakeEngagement', 'SteerChange'],
        ['ThrottlePercent', 'BrakeSwitch', 'SteerAngle']
    ):
        if pd.notna(idx):
            row = df.loc[idx]
            last5_speed = df.loc[
                (df['TrialTime'] > row['TrialTime'] - 5) &
                (df['TrialTime'] <= row['TrialTime'])
            ]['PlayerSpeed'].mean()

            temp_rows.append({
                'ParticipantID':        pid,
                'TrialNumber':          trial,
                'LightCondition':       row['LightCondition'],
                'CuesCondition':        row['CuesCondition'],
                'Condition':            row['Condition'],
                'ConditionTrialOrder':  row['ConditionTrialOrder'],
                'ResponseType':         label,
                'ResponseValue':        row[col],
                'ResponseTrialTime':    row['TrialTime'],
                'ResponseTimePostHazard': row['TrialTime'] - hazard_in_view_time,
                'DistanceToHazard':     row['DistanceBumperToBumper'],
                'PlayerDistanceToHazard': row['PlayerDistanceToHazard'],
                'Speed':                row['PlayerSpeed'],
                'MeanSpeedLast5s':      last5_speed,
                'PLDIntersectPlayerDist': pld_val,
                'PLDIntersectTime':     pld_time,
                'PLDIntersectSpeed':    pld_speed,
                'HazardAppearTime':     hazard_in_view_time,
                'MaxThrottle':          max_throttle_value,
                'MaxThrottleTime':      max_throttle_time,
                'ThrottleThreshold':    throttle_threshold,
                'SteerMean':            steer_mean,
                'SteerUpperThreshold':  steer_threshold_upper,
                'SteerLowerThreshold':  steer_threshold_lower,
            })

    trial_df = pd.DataFrame(temp_rows)
    if trial_df.empty:
        continue
    trial_df['ResponseTimeNum'] = pd.to_numeric(trial_df['ResponseTrialTime'], errors='coerce')
    trial_df = trial_df.sort_values('ResponseTimeNum', na_position='last')
    trial_df['ResponseOrder'] = trial_df.reset_index().groupby('TrialNumber').cumcount() + 1
    results.extend(trial_df.drop(columns='ResponseTimeNum').to_dict('records'))

response_summary = pd.DataFrame(results)

# Merge collision info
first_collision = (
    all_collisions
    .groupby(['ParticipantID', 'TrialNumber'])
    .agg(
        HazardCollision=('TrialTime', lambda x: len(x) > 0),
        FirstCollisionTime=('TrialTime', 'min')
    )
    .reset_index()
)
response_summary = pd.merge(
    response_summary, first_collision,
    on=['ParticipantID', 'TrialNumber'], how='left'
)
response_summary['HazardCollision'] = response_summary['HazardCollision'].fillna(False)
response_summary['DistFromPLD'] = (
    response_summary['PlayerDistanceToHazard'] - response_summary['PLDIntersectPlayerDist']
)

col_names = [
    'ParticipantID', 'TrialNumber', 'LightCondition', 'CuesCondition',
    'Condition', 'ConditionTrialOrder', 'ResponseType', 'ResponseOrder',
    'ResponseValue', 'ResponseTrialTime', 'ResponseTimePostHazard',
    'DistanceToHazard', 'PlayerDistanceToHazard', 'DistFromPLD',
    'Speed', 'MeanSpeedLast5s', 'HazardCollision',
    'PLDIntersectPlayerDist', 'PLDIntersectTime', 'PLDIntersectSpeed',
    'FirstCollisionTime', 'HazardAppearTime', 'MaxThrottle', 'MaxThrottleTime',
    'ThrottleThreshold', 'SteerMean', 'SteerUpperThreshold', 'SteerLowerThreshold'
]
response_summary = response_summary[col_names]
response_summary.to_csv(OUT_DIR / 'trial_responses.csv', index=False)
print(f"\nSaved trial_responses.csv  ({len(response_summary)} rows)")


# ============================================================
# 8. VISUALISE DETECTED RESPONSES (random sample of 5 trials)
# ============================================================

def plot_response_trials(hazard_data, response_summary, n_trials=5,
                          random_state=None, specific_trials=None):
    """Plot throttle / brake / steer time series with detected response lines."""
    if specific_trials is not None:
        sampled = pd.DataFrame(specific_trials, columns=['ParticipantID', 'TrialNumber'])
    else:
        sampled = (
            response_summary[['ParticipantID', 'TrialNumber']]
            .drop_duplicates()
            .sample(n=min(n_trials, response_summary[['ParticipantID', 'TrialNumber']]
                          .drop_duplicates().shape[0]),
                    random_state=random_state)
        )

    for _, info in sampled.iterrows():
        pid, trial = info['ParticipantID'], info['TrialNumber']
        df = (hazard_data[(hazard_data['ParticipantID'] == pid) &
                          (hazard_data['TrialNumber'] == trial)]
              .sort_values('TrialTime').copy())
        summary = response_summary[(response_summary['ParticipantID'] == pid) &
                                   (response_summary['TrialNumber'] == trial)]
        if summary.empty or df.empty:
            print(f"  Skipping {pid} trial {trial} — no data.")
            continue

        hazard_time      = summary['HazardAppearTime'].iloc[0]
        baseline_time    = summary['MaxThrottleTime'].iloc[0]
        throttle_thresh  = summary['ThrottleThreshold'].iloc[0]
        baseline_throttle = summary['MaxThrottle'].iloc[0]

        df['ThrottleSmooth'] = df['ThrottlePercent'].rolling(10, min_periods=1).mean()
        df['SteerSmooth']    = df['SteerAngle'].rolling(10, min_periods=1).mean()

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Participant {pid} — Trial {trial}", fontsize=14)

        ax = axes[0]
        ax.plot(df['TrialTime'], df['ThrottleSmooth'], color='tab:blue', label='Throttle (smooth)')
        ax.fill_between(df['TrialTime'], 65, 100, color='tab:blue', alpha=0.15, label='~65 mph zone')
        ax.axvline(hazard_time,    color='red',   linestyle='--', label='Hazard appears')
        ax.axvline(baseline_time,  color='green', linestyle='--', label='Max throttle')
        ax.axhline(baseline_throttle, color='green', linestyle=':')
        ax.axhline(throttle_thresh,   color='grey',  linestyle=':')
        row = summary[summary['ResponseType'] == 'ThrottleDecel']
        if not row.empty:
            ax.axvline(row['ResponseTrialTime'].iloc[0], color='grey',
                       linestyle='--', label='Throttle response')
        ax.set_ylabel('Throttle %')
        ax.set_ylim(-5, 105)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axes[1]
        ax.plot(df['TrialTime'], df['BrakeSwitch'], color='tab:orange', label='BrakeSwitch')
        ax.axvline(hazard_time, color='red', linestyle='--')
        row = summary[summary['ResponseType'] == 'BrakeEngagement']
        if not row.empty:
            ax.axvline(row['ResponseTrialTime'].iloc[0], color='grey',
                       linestyle='--', label='Brake response')
        ax.set_ylabel('Brake (0/1)')
        ax.set_ylim(-0.5, 1.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axes[2]
        ax.plot(df['TrialTime'], df['SteerSmooth'], color='tab:purple', label='SteerAngle (smooth)')
        ax.axvline(hazard_time, color='red', linestyle='--')
        row = summary[summary['ResponseType'] == 'SteerChange']
        if not row.empty:
            ax.axvline(row['ResponseTrialTime'].iloc[0], color='grey',
                       linestyle='--', label='Steer response')
        ax.set_ylabel('Steer Angle (°)')
        ax.set_xlabel('Trial Time (s)')
        ax.set_ylim(-40, 40)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = FIGURES_DIR / "response_checks" / f"response_check_{pid}_trial{trial}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path.name}")

print("\nGenerating response check plots…")
plot_response_trials(hazard_data, response_summary, n_trials=5, random_state=10)


# ============================================================
# 9. DATA ANALYSIS
# ============================================================

category_order = ['DayFull', 'DayNoCues', 'NightNoCues']
response_summary['Condition'] = pd.Categorical(
    response_summary['Condition'], categories=category_order, ordered=True
)
category_order1 = ['ThrottleDecel', 'BrakeEngagement', 'SteerChange']
response_summary['ResponseType'] = pd.Categorical(
    response_summary['ResponseType'], categories=category_order1, ordered=True
)
colors = {'DayFull': "#ff7f0e", 'DayNoCues': "#2ca02c", 'NightNoCues': "#1f77b4"}

# --- PLD stats ---
total_trials = response_summary.groupby(['ParticipantID', 'TrialNumber']).ngroups
valid_trials = (
    response_summary.dropna(subset=['DistFromPLD'])
    .groupby(['ParticipantID', 'TrialNumber'])
    .ngroups
)
print(f"\nPLD summary")
print(f"  Total unique trials:   {total_trials}")
print(f"  Trials with a PLD:     {valid_trials}  ({valid_trials/total_trials:.1%})")

pld_trial = (
    response_summary[['ParticipantID', 'TrialNumber', 'Condition', 'PLDIntersectPlayerDist']]
    .drop_duplicates(subset=['ParticipantID', 'TrialNumber'])
)
mean_pld = pld_trial.groupby('Condition')['PLDIntersectPlayerDist'].agg(['mean','std','count'])
print(f"\nPLD by condition:\n{mean_pld}")

# Repeated-measures ANOVA for PLD ~ Condition
aov = pg.rm_anova(
    dv='PLDIntersectPlayerDist', within='Condition',
    subject='ParticipantID', data=pld_trial, detailed=True
)
print(f"\nRM-ANOVA (PLD ~ Condition):\n{aov}")

posthoc = pg.pairwise_tests(
    dv='PLDIntersectPlayerDist', within='Condition',
    subject='ParticipantID', data=pld_trial
)
print(f"\nPost-hoc tests:\n{posthoc}")

# --- Summary bar charts ---
throttle_subset  = response_summary[response_summary['ResponseType'] == 'ThrottleDecel']
brake_subset     = response_summary[response_summary['ResponseType'] == 'BrakeEngagement']
steer_subset     = response_summary[response_summary['ResponseType'] == 'SteerChange']

def save_catplot(data, ylabel, title, out_name, 
                 subfolder="summary",
                 **kwargs):
    data = data.copy()
    if hasattr(data.get('Condition', None), 'cat'):
        data['Condition'] = data['Condition'].cat.remove_unused_categories()
    p = sns.catplot(data=data, kind='bar', palette=colors, **kwargs)
    p.set_axis_labels(kwargs.get('x', 'Condition'), ylabel)
    p.figure.suptitle(title, y=1.02)
    p.figure.savefig(FIGURES_DIR / subfolder / out_name, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {out_name}")

print("\nGenerating summary bar charts…")
save_catplot(response_summary.dropna(subset=['DistFromPLD']),
    ylabel='Mean Distance Between Actual and Theoretical Response (m)',
    title='Mean distance between actual response and PLD for each response type',
    out_name='dist_from_pld_by_response_type.png',
    x='ResponseType', y='DistFromPLD', hue='Condition', native_scale=False) 

# --- Mixed-effects regressions ---
def run_lmm(formula, data, group_col, label):
    model = smf.mixedlm(formula, data, groups=data[group_col])
    result = model.fit()
    print(f"\n{'='*60}")
    print(f"LMM: {label}")
    print(result.summary())
    return result

print("\nRunning mixed-effects models…")
throttle_lrf = run_lmm(
    'DistanceToHazard ~ Condition + MeanSpeedLast5s',
    throttle_subset, 'ParticipantID', 'Throttle: distance ~ condition + speed'
)
brake_lrf = run_lmm(
    'DistanceToHazard ~ Condition + MeanSpeedLast5s',
    brake_subset, 'ParticipantID', 'Brake: distance ~ condition + speed'
)
steer_lrf = run_lmm(
    'DistanceToHazard ~ Condition + MeanSpeedLast5s',
    steer_subset, 'ParticipantID', 'Steer: distance ~ condition + speed'
)

# First-trial regressions
first_throttle = response_summary[
    (response_summary['ResponseType'] == 'ThrottleDecel') &
    (response_summary['ConditionTrialOrder'] == 1)
]
first_brake = response_summary[
    (response_summary['ResponseType'] == 'BrakeEngagement') &
    (response_summary['ConditionTrialOrder'] == 1)
]
first_steer = response_summary[
    (response_summary['ResponseType'] == 'SteerChange') &
    (response_summary['ConditionTrialOrder'] == 1)
]

run_lmm('DistanceToHazard ~ Condition', first_throttle, 'ParticipantID',
        'Throttle (first trial): distance ~ condition')
run_lmm('DistanceToHazard ~ Condition', first_brake,   'ParticipantID',
        'Brake (first trial): distance ~ condition')
run_lmm('DistanceToHazard ~ Condition', first_steer,   'ParticipantID',
        'Steer (first trial): distance ~ condition')

# PLD distance regressions
pld_throttle = throttle_subset[['Condition', 'ParticipantID', 'DistFromPLD']].dropna()
pld_brake    = brake_subset[['Condition', 'ParticipantID', 'DistFromPLD']].dropna()
pld_steer    = steer_subset[['Condition', 'ParticipantID', 'DistFromPLD']].dropna()

run_lmm('DistFromPLD ~ Condition', pld_throttle, 'ParticipantID',
        'Throttle: dist-from-PLD ~ condition')
run_lmm('DistFromPLD ~ Condition', pld_brake,    'ParticipantID',
        'Brake: dist-from-PLD ~ condition')
run_lmm('DistFromPLD ~ Condition', pld_steer,    'ParticipantID',
        'Steer: dist-from-PLD ~ condition')

print(f"\nDone. All outputs written to: {OUT_DIR}")
