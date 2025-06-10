import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import time
import streamlit as st

def setup_cache():
    cache_dir = "f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

def get_race_list(year=2024):
    schedule = fastf1.get_event_schedule(year)
    return schedule[['RoundNumber', 'EventName']]

def load_race_data(round_number, year=2024):
    session = fastf1.get_session(year, round_number, "R")
    with st.spinner("🔄 Loading session data from FastF1..."):
        try:
            session.load()
        except Exception as e:
            st.error(f"❌ Failed to load session data: {e}")
            return pd.DataFrame()  # return empty to fail gracefully
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    return laps

def prepare_features(laps, qualifying_df, rain_prob, wet_perf, team_score, driver_mapping, driver_teams):
    sector_avg = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    sector_avg["TotalSectorTime (s)"] = sector_avg[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)

    qualifying_df["DriverCode"] = qualifying_df["Driver"].map(driver_mapping)
    qualifying_df["Team"] = qualifying_df["DriverCode"].map(driver_teams)
    qualifying_df["TeamPerformanceDelta"] = qualifying_df["Team"].map(team_score)

    merged = qualifying_df.merge(sector_avg, left_on="DriverCode", right_on="Driver", how="left")
    merged = merged.rename(columns={"Driver_x": "DriverFullName", "Driver_y": "DriverCodeMatched"})
    merged["WetPerformanceFactor"] = merged["DriverFullName"].map(wet_perf)
    merged["RainProbability"] = rain_prob

    avg_lap_times = laps.groupby("Driver")["LapTime (s)"].mean()
    merged = merged.merge(avg_lap_times, left_on="DriverCode", right_index=True, how="left")
    merged.dropna(subset=["LapTime (s)"], inplace=True)

    def apply_rain(row):
        if pd.isna(row["WetPerformanceFactor"]) or rain_prob == 0:
            return row["LapTime (s)"]
        factor = 0.11 if rain_prob == 0.5 else 0.52
        return row["LapTime (s)"] * (1 - factor + factor * row["WetPerformanceFactor"])

    merged["AdjustedLapTime (s)"] = merged.apply(apply_rain, axis=1)
    return merged
