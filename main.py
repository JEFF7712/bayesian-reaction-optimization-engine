import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

DATA_URL = "https://github.com/rxn4chemistry/rxn_yields/raw/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"

df = pd.read_excel(DATA_URL)

def smile_to_fp(smile, n_bits=2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)

ligand_fps = np.array([smile_to_fp(s) for s in df['Ligand']])
additive_fps = np.array([smile_to_fp(s) for s in df['Additive']])
base_fps = np.array([smile_to_fp(s) for s in df['Base']])
aryl_fps = np.array([smile_to_fp(s) for s in df['Aryl halide']])

X = np.hstack([ligand_fps, aryl_fps, base_fps, additive_fps])
y = df['Output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

def predict_with_uncertainty(model, X):
    per_tree_preds = [tree.predict(X) for tree in model.estimators_]
    per_tree_preds = np.vstack(per_tree_preds)
    mean_pred = np.mean(per_tree_preds, axis=0)
    std_pred = np.std(per_tree_preds, axis=0)
    
    return mean_pred, std_pred

y_pred_mean, y_pred_std = predict_with_uncertainty(rf, X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2 = r2_score(y_test, y_pred_mean)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}") 

def plot_data(y_test, y_pred_mean):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_mean, alpha=0.7)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel("Actual Yields")
    plt.ylabel("Predicted Yields")
    plt.title("Random Forest Regression: Actual vs Predicted Yields")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()

def acquisition_function(mean, std, kappa=2.0):
    return mean + (kappa * std)

# Example candidate pool using test set
candidate_pool = X_test

means, stds = predict_with_uncertainty(rf, candidate_pool)
scores = acquisition_function(means, stds, kappa=1.96)

best_idx = np.argmax(scores)
best_candidate_features = candidate_pool[best_idx]
predicted_yield = means[best_idx]
uncertainty = stds[best_idx]

print(f"Recommended Experiment ID: {best_idx}")
print(f"Predicted Yield: {predicted_yield:.2f}%")
print(f"Uncertainty: ±{uncertainty:.2f}%")
print(f"Acquisition Score:{scores[best_idx]:.2f}")

# Greedy choice - picking best mean over uncertainty (kappa=0)
greedy_idx = np.argmax(means)
print(f"\nvs Greedy Choice (Pure Exploitation):")
print(f"Predicted Yield: {means[greedy_idx]:.2f}%")
print(f"Uncertainty: ±{stds[greedy_idx]:.2f}%")