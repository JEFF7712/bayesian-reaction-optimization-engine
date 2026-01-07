import pandas as pd

DATA_URL = "https://github.com/rxn4chemistry/rxn_yields/raw/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"

df = pd.read_excel(DATA_URL)

print(df['Ligand'].unique())