import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

INPUT_PATH = Path("data/clean/dataset_clean_raw.csv")
OUTPUT_PATH = Path("data/processed/dataset_preprocessed.csv")

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

def procesar_texto(text: str, stop_words, lemmatizer) -> str:
    if not isinstance(text, str):
        return ""
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def main():
    setup_nltk()
    print(f"Cargando dataset limpio desde {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df["processed_text"] = df["clean_text"].apply(
        lambda x: procesar_texto(x, stop_words, lemmatizer)
    )

    # Filtrar textos demasiado cortos
    df = df[df["processed_text"].str.len() > 5]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset preprocesado guardado en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
