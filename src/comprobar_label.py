import pandas as pd
import os

# Nombre del archivo que buscamos
TARGET_FILE = "1_dataset_preprocessed.csv"

print(f"Buscando '{TARGET_FILE}' en el proyecto...")

archivo_encontrado = None

# Buscamos el archivo recursivamente por si está en una subcarpeta
for root, dirs, files in os.walk("."):
    if TARGET_FILE in files:
        archivo_encontrado = os.path.join(root, TARGET_FILE)
        break

if archivo_encontrado:
    print(f"--> ¡Archivo encontrado en!: {archivo_encontrado}")
    
    try:
        # Leemos el archivo
        df = pd.read_csv(archivo_encontrado, dtype=str) # Todo como texto para evitar errores
        
        # Normalizamos el nombre de la columna label (a veces viene como 'class', 'label', 'target')
        # Buscamos si existe 'label'
        if 'label' in df.columns:
            print("\n--- INFORME DE ETIQUETAS (LABEL) ---")
            conteo = df['label'].value_counts()
            print(conteo)
            
            # Análisis del resultado
            etiquetas_unicas = df['label'].unique()
            if len(etiquetas_unicas) > 1:
                print("\n[VITAL] ¡BUENAS NOTICIAS! Tienes clases mezcladas (0 y 1).")
                print("El problema era que estaban ordenadas o las borraste sin querer en el script anterior.")
            else:
                print(f"\n[PELIGRO] Solo tienes la etiqueta: {etiquetas_unicas[0]}")
                print("Este archivo NO sirve para entrenar. Necesitas buscar el dataset completo.")
        else:
            print(f"\n[ERROR] No encuentro la columna 'label'. Columnas disponibles: {list(df.columns)}")

    except Exception as e:
        print(f"Error al leer: {e}")
else:
    print(f"[ERROR] No encuentro '{TARGET_FILE}' en ninguna carpeta desde donde ejecutas el script.")