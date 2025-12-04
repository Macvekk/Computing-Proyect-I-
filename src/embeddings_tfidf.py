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
