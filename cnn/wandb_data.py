"""
Script to take csv and send to wandb to use their charting features
"""

import wandb
import pandas as pd

FILENAME = "wandb_export_2024-05-05T10_07_23.233-04_00.csv"
THRESHOLD = 1

loaded_experiment_df = pd.read_csv(FILENAME)

# grab column names from csv
# second arg in tuple represents the mapped name
METRIC_COLS = [
    ("imperial-droid-52 - Test Accuracy", "Test Accuracy"),
    ("imperial-droid-52 - Test Accuracy__MIN", ""),
    ("imperial-droid-52 - Test Accuracy__MAX", ""),
    ("imperial-droid-52 - Val Accuracy", "Val Accuracy"),
    ("imperial-droid-52 - Val Accuracy__MIN", ""),
    ("imperial-droid-52 - Val Accuracy__MAX", ""),
    ("imperial-droid-52 - Train Accuracy", "Train Accuracy"),
    ("imperial-droid-52 - Train Accuracy__MIN", ""),
    ("imperial-droid-52 - Train Accuracy__MAX", ""),
]

# initialize wandb
wandb.login()
run = wandb.init(project="cv-final-proj-data")
#
metrics = {}
for i, row in loaded_experiment_df.iterrows():
    for metric_col in METRIC_COLS:
        # if column is to be ignored or NaN, skip
        if metric_col[1] == "" or pd.isnull(row[metric_col[0]]):
            continue
        # apply multiplier to convert from decimal to percent
        multiplier = 100.0 if metric_col[1] == "Train Accuracy" else 1
        metrics[metric_col[1]] = row[metric_col[0]] * multiplier

    # if all metrics have been grabbed (train, val, test)
    if len(metrics) != THRESHOLD:
        continue

    run.log(metrics)
    metrics = {}  # reset dict
run.finish()
