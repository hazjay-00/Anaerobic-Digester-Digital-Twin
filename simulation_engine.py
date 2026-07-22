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
    S, X, M = y
    
    # Temperature correction factor (Arrhenius adaptation for bacteria kinetics)
    T_factor = np.exp(0.07 * (Temperature - 35.0))
    
    # Kinetic Parameters
    mu_max = 0.40 * T_factor  # Max growth rate scaled by temperature
    K_s = 180.0               # Half-velocity constant (mg/L)
    Y_xs = 0.45               # Yield coefficient (g biomass/g substrate)
    K_d = 0.02                # Bacterial decay rate
    Y_ms = 0.30               # Methane yield constant (L/mg)
    
    # Non-linear ODE Transformations
    dSdt = D * (S_inflow - S) - (mu_max * S / (K_s + S)) * X / Y_xs
    dXdt = (mu_max * S / (K_s + S)) * X - K_d * X - D * X
    dMdt = Y_ms * (mu_max * S / (K_s + S)) * X
    
    return [dSdt, dXdt, dMdt]

def run_plant_simulation(S_inflow, D, Temperature, days=10):
    """Runs a numerical trajectory for a single set of control inputs."""
    t = np.linspace(0, days, 24 * days) # Hourly points
    initial_conditions = [400.0, 80.0, 0.0] # [Initial Substrate, Initial Bacteria, Initial Methane]
    
    results = odeint(anaerobic_kinetics, initial_conditions, t, args=(S_inflow, D, Temperature))
    return results[-1, 0], results[-1, 1], results[-1, 2] # Return final [S, X, M] values

if __name__ == "__main__":
    print("Day 1 Physics Simulation Engine Compiled Successfully!")
    s, x, m = run_plant_simulation(500.0, 0.15, 37.0)
    print(f"Test Run Result -> Final Methane Production: {m:.2f} Liters")
# 