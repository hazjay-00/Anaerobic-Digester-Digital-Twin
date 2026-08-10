import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
from simulation_engine import run_plant_simulation

def generate_synthetic_industrial_dataset(samples=2000):
    """Simulates distinct operational permutations across full UI control limits."""
    print("Running Data Factory Monte Carlo Loop. Generating plant states...")
    np.random.seed(42)
    
    inflow_cod = np.random.uniform(150.0, 800.0, samples)     # Input Pollution (mg/L)
    dilution_rate = np.random.uniform(0.05, 0.30, samples)    # Dilution Rate (1/days)
    tank_temp = np.random.uniform(20.0, 60.0, samples)        # Inside Temp (°C)

    dataset = []
    for i in range(samples):
        s_out, x_out, methane_out = run_plant_simulation(inflow_cod[i], dilution_rate[i], tank_temp[i])
        dataset.append([inflow_cod[i], dilution_rate[i], tank_temp[i], s_out, x_out, methane_out])

    df = pd.DataFrame(dataset, columns=['Inflow_COD', 'Dilution_Rate', 'Temperature', 'Effluent_COD', 'Biomass_Density', 'Methane_Yield'])
    df.to_csv("simulated_plant_data.csv", index=False)
    print("Dataset manufactured successfully and saved to 'simulated_plant_data.csv'!")
    return df

def train_ai_operator_engine(df):
    """Trains the Machine Learning models on the synthetic data loops."""
    print("Organizing Data Partition Loops (80/20 Train-Test Framework)...")
    X = df[['Inflow_COD', 'Dilution_Rate', 'Temperature']]
    y_cod = df['Effluent_COD']

    X_train, X_test, y_train, y_test = train_test_split(X, y_cod, test_size=0.2, random_state=42)

    rf_cod_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_cod_model.fit(X_train, y_train)

    y_pred = rf_cod_model.predict(X_test)
    real_biogas_r2 = r2_score(y_test, y_pred) * 100.0

    print(f"ML Brain Training Complete. Test set COD prediction R² score: {real_biogas_r2:.2f}%")

    artifacts = {
        "model": rf_cod_model,
        "r2_score": real_biogas_r2
    }

    with open("twin_brain_cod.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("Model payload successfully written to 'twin_brain_cod.pkl'!")
    return real_biogas_r2

if __name__ == "__main__":
    df_sim = generate_synthetic_industrial_dataset()
    train_ai_operator_engine(df_sim)
