import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")