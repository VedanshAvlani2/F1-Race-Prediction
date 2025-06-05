"""
🏎️ Formula 1 Race Time Prediction Model (Enhanced Edition)
==========================================================
Predicts race lap times based on qualifying performance, historical data, 
weather conditions, and team performance using machine learning.

Features:
- Team performance scoring based on 2024 championship points
- Enhanced driver roster with 2025 lineup changes
- Comprehensive sector time analysis
- Weather impact modeling
"""

import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def setup_cache():
    """Initialize FastF1 cache directory"""
    cache_dir = "f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    print("✅ Cache directory initialized")

# ═══════════════════════════════════════════════════════════════════════════════
# 🏁 RACE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
# Load race schedule and user input
schedule = fastf1.get_event_schedule(2024)
race_list = schedule[['RoundNumber', 'EventName']].values.tolist()

print("Select a race by entering the corresponding round number:")
for race in race_list:
    print(f"{race[0]}: {race[1]}")
race_number = int(input("Enter the round number of the race: "))
selected_race = schedule[schedule['RoundNumber'] == race_number]['EventName'].values[0]
print(f"Selected race: {selected_race}")

# Load race session
session_2024 = fastf1.get_session(2024, race_number, "R")
session_2024.load()

# Lap time processing
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
sector_times_2024["TotalSectorTime (s)"] = sector_times_2024[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)

# Drivers and Qualifying Input
drivers = [
    "Oscar Piastri", 
    "George Russell", 
    "Lando Norris", 
    "Max Verstappen", 
    "Lewis Hamilton",
    "Charles Leclerc", 
    "Isack Hadjar", 
    "Andrea Kimi Antonelli", 
    "Yuki Tsunoda", 
    "Alexander Albon",
    "Esteban Ocon", 
    "Nico Hülkenberg", 
    "Fernando Alonso", 
    "Lance Stroll", 
    "Carlos Sainz Jr.",
    "Pierre Gasly", 
    "Oliver Bearman", 
    "Franco Colapinto", 
    "Gabriel Bortoleto", 
    "Liam Lawson"
]

qualifying_2025 = pd.DataFrame({
    "Driver": drivers,
    "QualifyingTime (s)": [float(input(f"Enter qualifying time for {driver} (in seconds): ")) for driver in drivers]
})

# Driver & team mapping
driver_mapping = {
    "Oscar Piastri": "PIA", 
    "George Russell": "RUS", 
    "Lando Norris": "NOR", 
    "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", 
    "Charles Leclerc": "LEC", 
    "Isack Hadjar": "HAD", 
    "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", 
    "Alexander Albon": "ALB", 
    "Esteban Ocon": "OCO", 
    "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", 
    "Lance Stroll": "STR", 
    "Carlos Sainz Jr.": "SAI", 
    "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", 
    "Franco Colapinto": "COL", 
    "Gabriel Bortoleto": "BOR", 
    "Liam Lawson": "LAW"
}

drivers_to_team = {
    "VER": "Red Bull", 
    "HAM": "Ferrari", 
    "LEC": "Ferrari", 
    "NOR": "McLaren", 
    "PIA": "McLaren", 
    "RUS": "Mercedes",
    "ALO": "Aston Martin", 
    "STR": "Aston Martin", 
    "SAI": "Williams", 
    "OCO": "Haas", 
    "GAS": "Alpine",
    "ALB": "Williams", 
    "HUL": "Kick Sauber", 
    "TSU": "Red Bull", 
    "HAD": "RB", 
    "ANT": "Mercedes",
    "BEA": "Haas", 
    "DOO": "Alpine", 
    "BOR": "Kick Sauber", 
    "LAW": "RB"
}

# Team standings to be updated after every race
team_points = {
    "Ferrari": 142, 
    "Mercedes": 147, 
    "McLaren": 319, 
    "Red Bull": 143,
    "Aston Martin": 14, 
    "RB": 22, 
    "Haas": 26, 
    "Williams": 54, 
    "Alpine": 7, 
    "Kick Sauber": 6
}

max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
qualifying_2025["Team"] = qualifying_2025["DriverCode"].map(drivers_to_team)
qualifying_2025["TeamPerformanceDelta"] = qualifying_2025["Team"].map(team_performance_score)

# Merge with sector data
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")
merged_data = merged_data.rename(columns={"Driver_x": "DriverFullName", "Driver_y": "DriverCodeMatched"})

# Add Wet Performance Factor
driver_wet_performance = {
    "Max Verstappen": 0.975196, 
    "Lewis Hamilton": 0.976464, 
    "Charles Leclerc": 0.975862,
    "Lando Norris": 0.978179, 
    "Fernando Alonso": 0.972655, 
    "George Russell": 0.968678,
    "Carlos Sainz Jr.": 0.978754, 
    "Yuki Tsunoda": 0.996338, 
    "Esteban Ocon": 0.981810,
    "Pierre Gasly": 0.978832, 
    "Lance Stroll": 0.979857
}
merged_data["WetPerformanceFactor"] = merged_data["DriverFullName"].map(driver_wet_performance)

# Weather input
rain_options = {"1": ("No Rain", 0), "2": ("Light Rain", 0.5), "3": ("Heavy Rain", 1)}
while True:
    print("\nSelect rain condition:")
    for k, v in rain_options.items():
        print(f"{k}. {v[0]}")
    rain_choice = input("Enter choice (1-3): ")
    if rain_choice in rain_options:
        _, rain_probability = rain_options[rain_choice]
        break
    print("Invalid input. Try again.")
merged_data["RainProbability"] = rain_probability

# Target Lap Time
avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean()
merged_data = merged_data.merge(avg_lap_times, left_on="DriverCode", right_index=True, how="left")
merged_data.dropna(subset=["LapTime (s)"], inplace=True)

def apply_rain_adjustment(row):
    if pd.isna(row["WetPerformanceFactor"]) or rain_choice == "1":
        return row["LapTime (s)"]
    factor = 0.11 if rain_choice == "2" else 0.52
    return row["LapTime (s)"] * (1 - factor + factor * row["WetPerformanceFactor"])
merged_data["AdjustedLapTime (s)"] = merged_data.apply(apply_rain_adjustment, axis=1)

# Features & Model
X = merged_data[[
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "RainProbability", "TeamPerformanceDelta"
]].fillna(0)
y = merged_data["AdjustedLapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict & Rank
merged_data["PredictedRaceTime (s)"] = model.predict(X)
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

# Results
print(f"\n\n🏁 Predicted 2025 {selected_race} Winner 🏁\n")
print(merged_data[["DriverFullName", "PredictedRaceTime (s)"]])

# Model performance
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_, edgecolor='black')
plt.xlabel("Importance Score")
plt.title("Feature Importance in Predicting Race Time")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
