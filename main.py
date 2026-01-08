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

# Convert SMILES to Morgan fingerprints
def smile_to_fp(smile, n_bits=2048):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=n_bits)
    mol = Chem.MolFromSmiles(smile)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp)

# Prepare features
ligand_fps = np.array([smile_to_fp(s) for s in df['Ligand']])
additive_fps = np.array([smile_to_fp(s) for s in df['Additive']])
base_fps = np.array([smile_to_fp(s) for s in df['Base']])
aryl_fps = np.array([smile_to_fp(s) for s in df['Aryl halide']])

# Split data into training set and candidate pool
X_full = np.hstack([ligand_fps, aryl_fps, base_fps, additive_fps])
y_full = df['Output'].values
initial_size = 10
X_train, X_pool, y_train, y_pool = train_test_split(X_full, y_full, train_size=initial_size, random_state=42)

# Lists to track performance for graph
history_best_yield = [np.max(y_train)]
history_avg_yield = [np.mean(y_train)]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Using custom prediction function to get std from RF
def predict_with_uncertainty(model, X):
    per_tree_preds = [tree.predict(X) for tree in model.estimators_]
    per_tree_preds = np.vstack(per_tree_preds)
    mean_pred = np.mean(per_tree_preds, axis=0)
    std_pred = np.std(per_tree_preds, axis=0)
    
    return mean_pred, std_pred

def acquisition_function(mean, std, kappa=2.0):
    return mean + (kappa * std)

n_iterations = 20

for i in range(n_iterations):
    # Train the model on current training data
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict on the unknown pool
    means, stds = predict_with_uncertainty(rf, X_pool)
    
    # Calculate acquisition scores
    scores = acquisition_function(means, stds, kappa=2.0)
    
    # Pick the winner/best candidate
    best_candidate_idx = np.argmax(scores)
    
    # Retrieve the actual yield from y_pool
    actual_yield = y_pool[best_candidate_idx]
    
    # Add new data to training set
    X_train = np.vstack([X_train, X_pool[best_candidate_idx]])
    y_train = np.append(y_train, actual_yield)
    
    # Remove discovered data from the pool
    X_pool = np.delete(X_pool, best_candidate_idx, axis=0)
    y_pool = np.delete(y_pool, best_candidate_idx, axis=0)
    
    # Logging
    current_max = np.max(y_train)
    history_best_yield.append(current_max)
    print(f"Step {i+1}/{n_iterations}: AI picked candidate with yield {actual_yield:.1f}%. (Global Best: {current_max:.1f}%)")

plt.figure(figsize=(10, 6))
plt.plot(range(len(history_best_yield)), history_best_yield, marker='o', linestyle='-', color='b', label='AI Optimization')
plt.axhline(y=np.max(y_full), color='r', linestyle='--', label='Theoretical Max (100%)')
plt.title('Optimization of Reaction Yields')
plt.xlabel('Number of Experiments Run')
plt.ylabel('Best Yield Found (%)')
plt.legend()
plt.grid(True)
plt.show()