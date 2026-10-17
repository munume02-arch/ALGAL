# Microalgae Biomass AI Web Dashboard

This is the web-based version of the algal biomass project using the provided 100-day dataset.

## What it does
- Shows an interactive experiment dashboard.
- Displays temperature, PAR, pH, OD680, nitrogen, CO2, dissolved oxygen, and biomass trends.
- Trains a small physics-informed neural network (PINN) on the 100-day dataset.
- Accepts current-day sensor values and estimates biomass in g/L.
- Accepts a CSV containing new sensor days and generates predicted biomass.
- Uses 5 model seeds and reports their average plus model spread.

## Run on Windows
1. Install Python.
2. Open Command Prompt in this folder.
3. Run:
   `python -m pip install -r requirements.txt`
4. Then run:
   `python -m streamlit run app.py`
5. Your browser should open the dashboard automatically. If not, copy the local address shown in Command Prompt into Chrome.

You can also double-click `run_dashboard.bat` after installing the requirements.

## Current prediction workflow
Enter the following parameters:

**Core Parameters (used by model):**
- Day
- Temperature (°C)
- PAR (µmol/m²/s)
- pH
- OD680

**Additional Environmental Parameters (for record):**
- Nitrogen (mg/L)
- CO2 (mg/L)
- Dissolved Oxygen (mg/L)

Then press **Predict Biomass**.

## Dataset Information
The supplied dataset contains 100 observations from one growth run spanning 100 days. The biomass grows from 0.08 g/L on Day 1 to 1.732 g/L on Day 100, showing a complete growth cycle including exponential and stationary phases. Additional parameters tracked include nitrogen levels (mg/L), CO2 concentration (mg/L), and dissolved oxygen (mg/L).

## Important research note
Predictions for new days, especially beyond Day 100 or under conditions outside the observed range, are estimates and should be validated experimentally. For a stronger FYP claim of robustness, collect multiple runs and include noisy/missing sensor conditions.
