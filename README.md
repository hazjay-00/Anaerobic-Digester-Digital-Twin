# Anaerobic Digester Digital Twin

A hybrid physics-ML simulation and control dashboard for continuous anaerobic bioreactors. This framework combines first-principles ODE kinetics (Monod & Haldane models) with data-driven machine learning to predict wastewater purification efficiency, methane yield, thermal stress boundaries, and real-time plant economics.


## System Architecture

The application is structured into three modular components:

1. **First-Principles Physics Engine (`simulation_engine.py`)**
   * Solves continuous mass-balance ordinary differential equations (ODEs) using `scipy.integrate.odeint`.
   * Integrates Arrhenius thermal scaling alongside Monod kinetic parameters for substrate degradation, active bacterial growth, and volumetric biogas generation.

2. **Empirical Machine Learning Layer (`ml_agent.py`)**
   * Executes a Monte Carlo generation loop across 2,000 synthetic operational permutations.
   * Trains a `RandomForestRegressor` to capture complex non-linearities and unmodeled real-world deviations in final effluent Chemical Oxygen Demand (COD).
   * Exports model parameters and validation metrics cleanly to `twin_brain_cod.pkl`.

3. **Interactive Control Dashboard (`interface.py`)**
   * Streamlit-based SCADA-style user interface.
   * Features operational safety checks, thermal acclimatization alerts, discharge compliance tracking, profit optimization baselines, and shift CSV report exports.
   * Includes an active emergency hardware kill-switch.
