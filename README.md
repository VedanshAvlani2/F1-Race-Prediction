# 🏎️ Predicting Formula 1 Race Times with Machine Learning

## 🧩 Overview
This project predicts Formula 1 race lap times using driver sector performance, qualifying sessions, team standings, and weather conditions. It integrates real-world data via the **FastF1** API and enhances accuracy using a **Gradient Boosting Regressor** model with historical team performance and wet-weather adaptability.
A Streamlit dashboard has been added to allow interactive selection of races, driver input, and real-time predictions.

## 🎯 Project Objective

- **Initial Goal**: Predict adjusted race lap times for drivers in upcoming GPs using relevant track and driver data.
- **Final Outcome**: Built a high-precision machine learning model that outputs a predicted finishing order and highlights key performance features. The model is now served through a web-based interface using Streamlit.

## 🗂️ Dataset & Inputs

This project pulls live and historical data from the [FastF1](https://theoehrly.github.io/Fast-F1/) Python library.

Key data components:
- **Driver Sector Times**: Sector 1, 2, and 3 averages from previous races.
- **Qualifying Times**: User-input qualifying time for each driver.
- **Team Performance Score**: Normalized constructor championship points.
- **Weather Effects**: Rain conditions adjusted using individual driver wet-weather performance factors.
- **Target Variable**: Adjusted race lap time.

## 🛠️ Technologies Used

- **Python**
- **FastF1** – Formula 1 data scraping
- **Scikit-learn** – Model training and evaluation
- **Streamlit** – Interactive dashboard
- **Matplotlib** – Visualization
- **Pandas, NumPy** – Data transformation

## 🚀 How to Run
You can run this project in two modes:

**⚙️ Option 1: Run Without the Dashboard (Script Mode)**
**If you prefer running the model from a Python script in Jupyter Notebook or VS Code:**
### 1. Clone the Repository & Install Requirements
```bash
pip install fastf1 pandas numpy matplotlib scikit-learn
```

### 2. Enable FastF1 Caching
On first run, FastF1 will prompt to download session data:
```python
fastf1.Cache.enable_cache("f1_cache")
```

### 3. Run the Model
Open the script in Jupyter or VS Code and execute:
```bash
python F1_Race_Predictor.py
```

You will be prompted to:
- Choose the race round (2024 season)
- Enter driver qualifying times
- Select rain condition (No rain / Light / Heavy)

**🖥️ Option 2: Run with the Streamlit Dashboard (Recommended)**
**If you want a web-based interactive experience:**
### 1. Clone the Repository & Install Requirements
```bash
pip install fastf1 pandas numpy matplotlib scikit-learn streamlit
```

### 2. Enable FastF1 Caching
On first run, FastF1 will prompt to download session data:
```python
fastf1.Cache.enable_cache("f1_cache")
```

### 3. Run the Model
Open the script in Jupyter or VS Code and execute:
```bash
streamlit run app.py
```
You will be able to:
- Select any completed race from the 2024 calendar
- Enter qualifying times for all drivers
- Choose weather conditions (No Rain / Light Rain / Heavy Rain)
- View predicted results, model accuracy, and feature importance live

## 🔁 Workflow

### 1. Data Integration
- Extracts 2024 race session lap data via FastF1
- Computes sector-wise averages per driver

### 2. Feature Engineering
- Integrates wet performance metrics per driver
- Applies rain-adjustment logic based on user input
- Maps constructor points into normalized team scores

### 3. Model Training
- Model: `GradientBoostingRegressor`
- Inputs: 7 engineered features
- Output: Adjusted predicted race lap time
- Validation: 80-20 train-test split

### 4. Prediction Output
- Drivers are ranked by predicted lap time
- Team score and gap from P1 shown
- Model accuracy displayed

## 📊 Results
**Model Metrics**:
- 🟢 **Mean Absolute Error**: 0.289 seconds
- 🧠 **Model Confidence**: High-precision predictions
- 🛠️ **Team Performance Factor**: Integrated as feature

## 🔍 Feature Importance
### Top Features:
- **Sector3Time (s)**: Most impactful predictor  
- **Sector2Time (s)** & **TeamPerformanceDelta**: Significant contributors  
- **RainProbability** & **WetPerformanceFactor**: Contextual but minor  

## 💡 Key Takeaways

- Sector performance is a stronger predictor than qualifying time alone.
- Constructor performance has a tangible impact on predicted outcomes.
- The model adapts for rain using driver-specific wet weather profiles.
- A dashboard-based approach allows fast iteration, testing, and visualization of race scenarios.

## 📈 Future Enhancements

- Add live race-day telemetry data (pit stops, tire degradation)
- Simulate race incidents like safety cars and DNFs
- Visualize full-race simulations with animated lap-by-lap results
- 🔧 Enable toggling between live FastF1 data and offline fallback mode
- Export predictions to CSV or PDF
- Add scenario simulation controls for pit stop strategies and tire compounds
