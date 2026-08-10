import numpy as np
import pandas as pd
from scipy.integrate import odeint

def anaerobic_kinetics(y, t, S_inflow, D, Temperature):
    """
    Solves continuous bioreactor mass balances.
    y[0] = S (Substrate/Pollution concentration in mg/L)
    y[1] = X (Biomass/Active Bacteria concentration in mg/L)
    y[2] = M (Accumulated Biogas/Methane yield in Liters)
    """
    S = max(0.0, y[0])
    X = max(0.0, y[1])
    M = max(0.0, y[2])

    # Temperature correction factor (Arrhenius adaptation + upper thermal decay boundary)
    if Temperature > 45.0:
        T_factor = np.exp(0.07 * (45.0 - 35.0)) * np.exp(-0.15 * (Temperature - 45.0))
    else:
        T_factor = np.exp(0.07 * (Temperature - 35.0))

    # Kinetic Parameters
    mu_max = 0.40 * T_factor   # Max growth rate scaled by temperature
    K_s = 180.0                # Half-velocity constant (mg/L)
    Y_xs = 0.45                # Yield coefficient (g biomass/g substrate)
    K_d = 0.02                 # Bacterial decay rate
    Y_ms = 0.30                # Methane yield constant (L/mg)

    # Specific growth rate calculation
    mu = mu_max * S / (K_s + S) if (K_s + S) > 0 else 0.0

    # Non-linear ODE Transformations
    dSdt = D * (S_inflow - S) - (mu * X / Y_xs)
    dXdt = mu * X - K_d * X - D * X
    dMdt = Y_ms * mu * X

    return [dSdt, dXdt, dMdt]

def run_plant_simulation(S_inflow, D, Temperature, days=10):
    """Runs a numerical trajectory for a single set of control inputs."""
    t = np.linspace(0, days, 24 * days)  # Hourly points
    initial_conditions = [400.0, 80.0, 0.0]  # [Initial Substrate, Initial Bacteria, Initial Methane]
    
    results = odeint(anaerobic_kinetics, initial_conditions, t, args=(S_inflow, D, Temperature))
    
    # Ensure returned final states cannot be negative
    S_final = max(0.0, results[-1, 0])
    X_final = max(0.0, results[-1, 1])
    M_final = max(0.0, results[-1, 2])
    
    return S_final, X_final, M_final

if __name__ == "__main__":
    print("Day 1 Physics Simulation Engine Compiled Successfully!")
    s, x, m = run_plant_simulation(500.0, 0.15, 37.0)
    print(f"Test Run Result -> Substrate: {s:.2f} mg/L, Biomass: {x:.2f} mg/L, Methane: {m:.2f} L")
