"""
Builds the hourly Paranal-specific training dataset by merging three real
ESO instrument archives on a common hourly timestamp:

  - seeing.csv         DIMM (Differential Image Motion Monitor): real seeing (target)
  - lhatpro.csv        LHATPRO microwave radiometer: water vapour / sky temperature
  - meteo_YYYY.csv      30 m meteorological tower (one file per year; see README
                        for why the raw export is split by year)

All three instruments log at different, irregular native cadences (roughly
1-2 minutes). Each is independently aggregated to hourly statistics first
(mean for most quantities; circular mean for wind direction; standard
deviation, not mean, for vertical wind speed, since its mean is ~0 but its
variance is a genuine turbulence proxy), and then inner-joined on the shared
hourly timestamp. This guarantees exact temporal alignment across sources.

Run:
    python src/build_dataset.py

Expects seeing.csv, lhatpro.csv, and meteo_<year>.csv (2016-2026) in the
current working directory. Produces paranal_specific_dataset.csv.
"""

import numpy as np
import pandas as pd

SEEING_FILE = "seeing.csv"
LHATPRO_FILE = "lhatpro.csv"
METEO_FILES = [f"meteo_{y}.csv" for y in range(2016, 2027)]

OUTPUT_FILE = "paranal_specific_dataset.csv"

# Final feature set for this Paranal-specific model.
# Note: humidity_2m is intentionally excluded here even though it is computed
# below - see train_dnn.py / train_xgboost.py FEATURE_ORDER, where it was
# dropped after permutation importance showed a slightly *negative* value
# (pure noise). It is kept in this merge script's output for completeness;
# training scripts decide the final feature subset.
FEATURE_ORDER = [
    "airmass",
    "pressure",
    "temp_2m", "temp_30m", "temp_ground",
    "temp_gradient_ground_30m",
    "dew_point_depression_2m",
    "humidity_2m",
    "wind_speed_10m", "wind_speed_30m",
    "wind_shear_10_30",
    "wind_dir_10m",
    "wind_w_std",
    "rain_intensity",
    "ir_temperature", "liquid_water_path", "pwv",
]
TARGET = "seeing"


def circular_mean_deg(series):
    """Correct mean for a wind-direction angle (359° and 1° are close in
    reality but far apart as raw numbers; a plain arithmetic mean would be
    wrong)."""
    radians = np.deg2rad(series.dropna())
    if len(radians) == 0:
        return np.nan
    mean_angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return (np.rad2deg(mean_angle) + 360) % 360


def load_seeing():
    df = pd.read_csv(SEEING_FILE, comment="#")
    df["datetime"] = pd.to_datetime(df["Date time"], utc=True)
    df["hour"] = df["datetime"].dt.floor("h")
    hourly = df.groupby("hour").agg(
        seeing=('DIMM Seeing ["]', "mean"),
        airmass=("Airmass", "mean"),
    ).reset_index().rename(columns={"hour": "datetime"})
    return hourly


def load_lhatpro():
    df = pd.read_csv(LHATPRO_FILE, comment="#")
    df["datetime"] = pd.to_datetime(df["Date time"], utc=True)
    df["hour"] = df["datetime"].dt.floor("h")
    hourly = df.groupby("hour").agg(
        ir_temperature=("IR temperature [Celsius]", "mean"),
        liquid_water_path=("Liquid water path [g/m**2]", "mean"),
        pwv=("Precipitable Water Vapour [mm]", "mean"),
    ).reset_index().rename(columns={"hour": "datetime"})
    return hourly


def load_meteo():
    """
    Processes one yearly meteo file at a time (aggregate to hourly, then
    discard the raw ~1-minute data) rather than concatenating ~5 million raw
    rows across 11 years in memory at once.
    """
    hourly_parts = []
    for path in METEO_FILES:
        try:
            df = pd.read_csv(path, comment="#")
        except FileNotFoundError:
            print(f"   [skip] {path} not found")
            continue

        df["datetime"] = pd.to_datetime(df["Date time"], utc=True)
        df["hour"] = df["datetime"].dt.floor("h")

        grouped = df.groupby("hour")
        part = grouped.agg(
            pressure=("Air Pressure at ground [hPa]", "mean"),
            temp_2m=("Air Temperature at 2m [C]", "mean"),
            temp_30m=("Air Temperature at 30m [C]", "mean"),
            temp_ground=("Air Temperature at ground [C]", "mean"),
            dew_2m=("Dew Temperature at 2m [C]", "mean"),
            humidity_2m=("Relative Humidity at 2m [%]", "mean"),
            wind_speed_10m=("Wind Speed at 10m [m/s]", "mean"),
            wind_speed_30m=("Wind Speed at 30m [m/s]", "mean"),
            wind_w_std=("Wind Speed W at 20m [m/s]", "std"),
            rain_intensity=("Rain intensity below VLT [%]", "mean"),
        ).reset_index().rename(columns={"hour": "datetime"})

        wind_dir = grouped["Wind Direction at 10m (0/360) [deg]"].apply(circular_mean_deg)
        part["wind_dir_10m"] = wind_dir.values

        hourly_parts.append(part)
        print(f"   {path}: {len(part)} hours (aggregated, raw data freed)")
        del df

    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly.drop_duplicates(subset=["datetime"])

    # Physically motivated derived features
    hourly["temp_gradient_ground_30m"] = hourly["temp_ground"] - hourly["temp_30m"]
    hourly["dew_point_depression_2m"] = hourly["temp_2m"] - hourly["dew_2m"]
    hourly["wind_shear_10_30"] = hourly["wind_speed_30m"] - hourly["wind_speed_10m"]

    return hourly.drop(columns=["dew_2m"])


def build_dataset():
    print("Loading and hourly-aggregating seeing.csv ...")
    seeing = load_seeing()
    print(f"  {len(seeing)} hours")

    print("Loading and hourly-aggregating lhatpro.csv ...")
    lhatpro = load_lhatpro()
    print(f"  {len(lhatpro)} hours")

    print("Loading and hourly-aggregating meteo_*.csv ...")
    meteo = load_meteo()
    print(f"  {len(meteo)} hours")

    # Exact inner join on shared hourly timestamp: only hours with data from
    # all three instruments are kept.
    merged = seeing.merge(lhatpro, on="datetime", how="inner")
    merged = merged.merge(meteo, on="datetime", how="inner")

    cols = ["datetime"] + FEATURE_ORDER + [TARGET]
    merged = merged[cols].dropna(subset=FEATURE_ORDER + [TARGET])

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(merged)} final rows (seeing + lhatpro + meteo aligned) to {OUTPUT_FILE}")
    print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")


if __name__ == "__main__":
    build_dataset()
