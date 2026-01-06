import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle

INPUT_PATH = Path("data/processed/dataset_preprocessed.csv")
OUTPUT_FEATURES_PATH = Path("data/features_tfidf.pkl")

def main():
    print(f"Cargando dataset preprocesado desde {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    if "processed_text" not in df.columns:
        raise ValueError("La columna 'processed_text' no existe en el dataset.")

    if "label" not in df.columns:
        raise ValueError("La columna 'label' no existe en el dataset. Asegúrate de tener las etiquetas.")

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["label"]

    OUTPUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURES_PATH, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Embeddings TF-IDF guardados en {OUTPUT_FEATURES_PATH}")

if __name__ == "__main__":
    main()

"""
Anotaciones sobre que hacer en embeddings_tfidf.py

Cargar el CSV Limpio
Configurar el vectorizado TF-IDF
Generar la matriz de embeddings
Guardar a disco los embeddings TF-IDF
Mostrar estadisticas bácicas (dimensaiones, vocabulario, etc.)

Decisiones TF-IDF son:
Parametro    Decision    Motivo
stop_words  'english'   Eliminar palabras comunes en inglés que no aportan significado relevante al análisis.
min_df      2-5         Eliminar términos raros
ngram_range (1,2)       Incluir tanto unigramas como bigramas para capturar más contexto en el texto.
sublinear_tf True        Aplicar escala sublineal para reducir el impacto de términos muy frecuentes. (Penaliza Repeticiones)
max_features   opcional  Controlar el tamaño del vocabulario para manejar la dimensionalidad y mejorar la eficiencia computacional.

Responsabilidad unica del script: Convertir texto limpio en embeddings TF-IDF y guardarlos para su uso posterior.

Ejemplo de código para cargar los embeddings guardados:
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# =========================
# 1. RUTAS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "reddit_depression_clean.csv"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "tfidf_embeddings.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

# =========================
# 2. CARGAR DATOS LIMPIOS
# =========================

df = pd.read_csv(DATA_PATH)
print("Dataset limpio cargado")

texts = df["clean_text"].tolist()

# =========================
# 3. TF-IDF VECTORIZER
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# =========================
# 4. GENERAR EMBEDDINGS
# =========================

X_tfidf = vectorizer.fit_transform(texts)

print("Embeddings TF-IDF generados")
print(f"Forma de la matriz TF-IDF: {X_tfidf.shape}")

# =========================
# 5. GUARDAR RESULTADOS
# =========================

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(X_tfidf, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Embeddings y vectorizador guardados")

# =========================
# 6. INSPECCIÓN BÁSICA
# =========================

vocab_size = len(vectorizer.get_feature_names_out())
print(f"Tamaño del vocabulario: {vocab_size}")

Diferencias entre los scripts de TF-IDF y limpieza de datos:
data_cleaning.py --> Limpieza estrcucturada del texto crudo y linguistica.
embeddings_tfidf.py --> Generación de embeddings TF-IDF a partir del texto limpio, representacion vectorial.
model.py --> Entrenamiento y evaluación de modelos ML usando los embeddings generados. clasificación/clustering.

Importante no mezclar responsabilidades entre estos scripts para mantener un código limpio y modular.

Se puede añadir si se considera:
    - comparativa TF vs TF-IDF.
    - Analisis de palabras más relevantes por clase.
    - Visualizacion PCA/t-SNE (No Modelado)

"""