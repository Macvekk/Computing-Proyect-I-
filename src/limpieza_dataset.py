import pandas as pd
import re
from pathlib import Path

# Ruta al dataset original
INPUT_PATH = Path("data/raw/dataset.csv")
OUTPUT_PATH = Path("data/clean/dataset_clean_raw.csv")

def limpiar_texto(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # URLs
    text = re.sub(r"@\w+", "", text)          # menciones
    text = re.sub(r"#[A-Za-z0-9_]+", "", text) # hashtags
    text = re.sub(r"[^\w\s]", " ", text)      # signos raros
    text = re.sub(r"\s+", " ", text)          # espacios múltiples
    return text.strip()

def main():
    print(f"Cargando dataset desde {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    # Quitar duplicados y nulos
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["text"], inplace=True)

    # Aplicar limpieza
    df["clean_text"] = df["text"].apply(limpiar_texto)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset limpio guardado en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
