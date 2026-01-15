"""
Script 3: Vectorización TF-IDF
Asignatura: Computación 1
Descripción: Convierte el texto procesado en matriz numérica (Sparse Matrix) y guarda el modelo en ficheros pickle (.pkl).
"""

import sys
import pickle
import time
import pandas as pd
import numpy as np  # se ha añadido esta línea para evitar los errores de NameError
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. CONFIGURACIÓN Y CONSTANTES

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# FICHERO DE ENTRADA
INPUT_FILE = DATA_DIR / "3_dataset_preprocess.csv"
# FICHERO DE SALIDA
OUTPUT_FILE = DATA_DIR / "features_tfidf.pkl"

TEXT_COLUMN = "processed_text"
LABEL_COLUMN = "label"

# CONFIGURACIÓN TF-IDF
TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "dtype": np.float32 # AQUI USO EL NUMPY IMPORTADO
}

# 2. CLASE VECTORIZADORA

class TfidfPipeline:
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(**config)

    def fit_transform(self, texts):
        """Aprendiendo el vocabulario incliuido. Transforma el texto a vectores."""
        n_docs = len(texts)
        print(f"Iniciando vectorización de {n_docs:,} documentos...")
        print("Se paciente, estamos tratando más de 1.5 milllones de filas, puede tardar unos minutos. No cierres el proceso.")      
        start_time = time.time()
        
        try:
            # Se ha intentado incluir una barra de progreso pero no se puede para fit_transform.
            matrix = self.vectorizer.fit_transform(texts)
            
            elapsed = time.time() - start_time
            print(f"Vectorización terminada en {elapsed:.2f} segundos.")
            return matrix
            
        except MemoryError: # En caso de error de memoria, por ejemplo, si se lanza en un colab.
            print("\n[ERROR FATAL] ¡Memoria insuficiente (RAM)!")
            print("Intenta reducir 'max_features' a 2000 o usa una muestra del dataset.")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Fallo en vectorización: {e}")
            sys.exit(1)

    def save_artifacts(self, matrix, labels, output_path):
        """Guarda X (matriz), y (labels) y el propio vectorizer en un .pkl"""
        
        artifacts = {
            "X": matrix,              
            "y": labels,              
            "vectorizer": self.vectorizer,  
            "feature_names": self.vectorizer.get_feature_names_out() 
        }

        print(f"Guardando pickle en disco ({output_path.name})...")
        try:
            with open(output_path, "wb") as f:
                pickle.dump(artifacts, f)
            return True
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el pickle: {e}")
            return False

# 3. PIPELINE PRINCIPAL

def run_tfidf_pipeline():
    print("--- Iniciando Pipeline TF-IDF ---")

    # 1. Cargar Datos
    if not INPUT_FILE.exists():
        print(f"[ERROR] No se encuentra: {INPUT_FILE}")
        sys.exit(1)

    print(f"1. Cargando dataset desde: {INPUT_FILE.name}")
    # Uso low_memory=False para evitar warnings en cargas grandes
    df = pd.read_csv(INPUT_FILE, dtype={LABEL_COLUMN: str}, low_memory=False)

    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Faltan columnas requeridas ({TEXT_COLUMN} o {LABEL_COLUMN})")

    # Eliminar nulos de seguridad
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    
    texts = df[TEXT_COLUMN].astype(str).tolist()
    labels = df[LABEL_COLUMN].tolist()

    print(f"   Registros listos: {len(texts):,}")

    # 2. Vectorización
    print("2. Generando Matriz TF-IDF...")
    pipeline = TfidfPipeline(TFIDF_CONFIG)
    
    # Este paso es el que demora en completar, especialmente con datasets grandes.
    X = pipeline.fit_transform(texts)
    
    n_samples, n_features = X.shape
    print(f"Matriz generada: {n_samples} filas x {n_features} columnas")

    # 3. Guardado de Pkl
    print("3. Guardando archivo pickle (.pkl)...")
    success = pipeline.save_artifacts(X, labels, OUTPUT_FILE)

    if success:
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print("\n" + "="*40)
        print(" PROCESO COMPLETADO EXITOSAMENTE")
        print("="*40)
        print(f"Archivo: {OUTPUT_FILE.name}")
        print(f"Tamaño:  {size_mb:.2f} MB")
        print("="*40)

    # 4. Main
if __name__ == "__main__":
    run_tfidf_pipeline()