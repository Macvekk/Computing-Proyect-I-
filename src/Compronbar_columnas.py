import pandas as pd
import os

# Ajusta la ruta si es necesario
ruta = "c:/eclipse/Proyecto_Computación_1/Computing-Proyect-I-/src/data/3_dataset_preprocess.csv"

if os.path.exists(ruta):
    df = pd.read_csv(ruta)
    print("\n--- COLUMNAS ENCONTRADAS ---")
    print(list(df.columns))
    print("----------------------------\n")
else:
    print("No encuentro el archivo para revisar.")