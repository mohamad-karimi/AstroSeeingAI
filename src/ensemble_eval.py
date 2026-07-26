"""
Exploratory script: evaluates a weighted ensemble of the DNN (train_dnn.py)
and XGBoost (train_xgboost.py) predictions on the identical chronological
test set, across a sweep of blend weights.

Conclusion (see docs/report_EN.docx, Section 4.5 / Figure 3): the best blend
(0.8 x XGBoost + 0.2 x DNN) reaches R2 = 0.6988, only 0.0025 above XGBoost
alone (0.6963) - within the noise level of this evaluation. This script is
kept for transparency and reproducibility, but the ensemble was NOT adopted
as the production model; train_xgboost.py alone is the final model.

Run:
    python src/ensemble_eval.py

Requires dnn_model.keras, xgb_model.json, and paranal_specific_dataset.csv
in the current working directory.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from xgboost import XGBRegressor

DATA_FILE = "paranal_specific_dataset.csv"
TARGET = "seeing"

DNN_FEATURES = [
    "airmass", "pressure",
    "temp_2m", "temp_30m", "temp_ground",
    "temp_gradient_ground_30m",
    "dew_point_depression_2m",
    "wind_speed_10m", "wind_speed_30m",
    "wind_shear_10_30",
    "wind_dir_10m",
    "wind_w_std",
    "rain_intensity",
    "ir_temperature", "liquid_water_path", "pwv",
]

XGB_FEATURES = [
    "airmass", "pressure",
    "temp_2m", "temp_30m", "temp_ground",
    "temp_gradient_ground_30m",
    "dew_point_depression_2m",
    "wind_speed_10m", "wind_speed_30m",
    "wind_shear_10_30",
    "wind_dir_sin", "wind_dir_cos",
    "wind_w_std",
    "rain_intensity",
    "ir_temperature", "pwv",
    "temp_2m_roll3h_mean", "temp_2m_trend_3h",
    "wind_speed_10m_roll3h_mean", "wind_speed_10m_trend_3h",
    "wind_speed_30m_roll3h_mean", "wind_speed_30m_trend_3h",
    "pressure_trend_3h", "wind_w_std_roll3h_mean",
]

# -------------------- Load + rebuild engineered features --------------------
dataset = pd.read_csv(DATA_FILE, encoding="utf-8")
dataset = dataset.sort_values("datetime").reset_index(drop=True)

wind_rad = np.deg2rad(dataset["wind_dir_10m"])
dataset["wind_dir_sin"] = np.sin(wind_rad)
dataset["wind_dir_cos"] = np.cos(wind_rad)

for col in ["temp_2m", "wind_speed_10m", "wind_speed_30m"]:
    dataset[f"{col}_roll3h_mean"] = dataset[col].rolling(3, min_periods=1).mean()
    dataset[f"{col}_trend_3h"] = dataset[col] - dataset[col].shift(3)

dataset["pressure_trend_3h"] = dataset["pressure"] - dataset["pressure"].shift(3)
dataset["wind_w_std_roll3h_mean"] = dataset["wind_w_std"].rolling(3, min_periods=1).mean()

all_needed_cols = sorted(set(DNN_FEATURES) | set(XGB_FEATURES) | {TARGET})
dataset = dataset.dropna(subset=all_needed_cols)
dataset["log_seeing"] = np.log1p(dataset[TARGET])

# -------------------- Identical chronological split (must match train_dnn.py / train_xgboost.py) --------------------
dataset["week_block"] = pd.to_datetime(dataset["datetime"], utc=True).dt.to_period("W").astype(str)
blocks_in_order = dataset["week_block"].drop_duplicates().tolist()
n_blocks = len(blocks_in_order)
n_test_blocks = max(1, round(n_blocks * 0.15))
test_blocks = set(blocks_in_order[-n_test_blocks:])

test_dataset = dataset[dataset["week_block"].isin(test_blocks)]
y_test_real = test_dataset[TARGET].copy()

# -------------------- Predictions --------------------
dnn_model = keras.models.load_model("dnn_model.keras")
x_test_dnn = test_dataset[DNN_FEATURES].select_dtypes(include=np.number)
dnn_pred = np.expm1(dnn_model.predict(x_test_dnn, verbose=0).flatten())

xgb_model = XGBRegressor()
xgb_model.load_model("xgb_model.json")
x_test_xgb = test_dataset[XGB_FEATURES].select_dtypes(include=np.number)
xgb_pred = np.expm1(xgb_model.predict(x_test_xgb))


def report(name, pred):
    mae = mean_absolute_error(y_test_real, pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred))
    r2 = r2_score(y_test_real, pred)
    print(f"{name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return r2


report("DNN", dnn_pred)
report("XGBoost", xgb_pred)

for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    ensemble_pred = w * xgb_pred + (1 - w) * dnn_pred
    report(f"Ensemble(w={w})", ensemble_pred)

print(
    "\nConclusion: best ensemble weight (~0.8) improves R2 by ~0.0025 over "
    "XGBoost alone - within noise. XGBoost alone is used as the production "
    "model (see train_xgboost.py)."
)
