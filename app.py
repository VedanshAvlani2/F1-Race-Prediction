import streamlit as st
import pandas as pd
from model_logic import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# App Title
st.title("🏁 F1 Predictive Analytics Dashboard")

# Setup
setup_cache()
race_df = get_race_list()
race_map = {f"{r} - {n}": r for r, n in race_df.values}

# Race Selection
race_select = st.selectbox("Select Race", list(race_map.keys()))
race_round = race_map[race_select]

# Qualifying Input
st.subheader("📊 Enter Qualifying Times (in seconds)")
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
qual_times = {}

for driver in drivers:
    label = str(driver).encode('ascii', 'ignore').decode()  # removes special characters like ü, é, etc.
    qual_times[driver] = st.number_input(label, value=90.0, step=0.001, format="%.3f")

qualifying_df = pd.DataFrame({
    "Driver": drivers,
    "QualifyingTime (s)": [qual_times[d] for d in drivers]
})

# Rain
st.subheader("🌧️ Select Weather Condition")
rain_choice = st.radio("Rain", ["No Rain", "Light Rain", "Heavy Rain"])
rain_map = {"No Rain": 0, "Light Rain": 0.5, "Heavy Rain": 1}
rain_prob = rain_map[rain_choice]

# Button to Predict
if st.button("🚀 Predict Race Results"):
    laps = load_race_data(race_round)

    # Dictionaries (your same mappings from earlier)
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
    driver_teams = {
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
    
    wet_perf = {
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
    max_pts = max(team_points.values())
    team_score = {t: p/max_pts for t, p in team_points.items()}

    df = prepare_features(laps, qualifying_df, rain_prob, wet_perf, team_score, driver_mapping, driver_teams)

    # Train model
    X = df[[
        "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
        "WetPerformanceFactor", "RainProbability", "TeamPerformanceDelta"
    ]].fillna(0)
    y = df["AdjustedLapTime (s)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
    model.fit(X_train, y_train)
    df["PredictedRaceTime (s)"] = model.predict(X)
    df = df.sort_values("PredictedRaceTime (s)")

    # Results
    st.success(f"🏆 Predicted Winner: {df.iloc[0]['DriverFullName']}")
    st.dataframe(df[["DriverFullName", "PredictedRaceTime (s)"]].reset_index(drop=True))

    # Feature importance
    st.subheader("🔍 Feature Importance")
    fig, ax = plt.subplots()
    ax.barh(X.columns, model.feature_importances_, edgecolor='black')
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    st.pyplot(fig)

    # Model error
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"🔍 MAE: {mae:.2f} seconds")
