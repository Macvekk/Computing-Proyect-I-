"""import pandas as pd
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
    main()"""


from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#------------------------------------------------
#CONFIGURACIÓN
#------------------------------------------------

INPUT_PATH = Path("data/clean/dataset_clean_raw.csv")  # entrada
OUTPUT_PATH = Path("data/processed/dataset_preprocessed.csv") # salida
TEXT_COLUMN = "clean_text" # Columna de texto de entrada
OUTPUT_TEXT_COLUMN = "processed_text" # Columna de texto de salida


# Config global para el pipeline de limpieza.

@dataclass
class TextCleaningConfig:
   
    language: str = "english"   # "english", "spanish", etc. (según NLTK)
    min_doc_length: int = 5     # Longitud mínima del texto final (caracteres)
    min_token_length: int = 2   # Longitud mínima de cada token
    remove_numbers: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True  # @usuario
    to_lowercase: bool = True

#------------------------------------------------
#NTLK SETUP
#------------------------------------------------

def setup_nltk(language: str = "english") -> None: # Descarga recursos necesarios de NLTK si no están
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
# Si cambiamos a otro idioma, NLTK tiene qye tener las stopwords en ese idioma
    stopwords.words(language)  # fuerza la descarga si falta

#------------------------------------------------
#FUNCIONES DE LIMPIEZA
#------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")  # caracteres no ASCII



def normalize_text(text: str, cfg: TextCleaningConfig) -> str:
#Aplica limpieza básica usando regex (antes del tokenizado).
    if not isinstance(text, str):
        return ""

    # minúsculas
    if cfg.to_lowercase:
        text = text.lower()

    # Quitado URLs
    if cfg.remove_urls:
        text = URL_PATTERN.sub(" ", text)

    # Quitar menciones tipo @usuario (por ejemplo, en Twitter)
    if cfg.remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)

    # Reemplazar saltos de línea y tabulaciones por espacios
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Eliminacion caracteres no ASCII
    
    text = NON_ASCII_PATTERN.sub(" ", text)

    # Compactar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_tokens(
    tokens: Iterable[str],
    stop_words: set[str],
    cfg: TextCleaningConfig,
) -> list[str]:
# Aplica filtros a nivel de token: alfabéticos, stopwords, longitud mínima, números."""
    cleaned_tokens: list[str] = []

    for t in tokens:
        # Eliminar tokens que no sean alfabéticos (puntuación, símbolos, etc.)
        if not t.isalpha():
            if cfg.remove_numbers:
                # si t es número o mezcla o símbolo, lo saltamos
                continue

        # Eliminar stopwords
        if t in stop_words:
            continue

        # Longitud mínima de token
        if len(t) < cfg.min_token_length:
            continue

        cleaned_tokens.append(t)

    return cleaned_tokens

# --------------------------------------------------------------------
# CLASE PRINCIPAL DEL PIPELINE
# --------------------------------------------------------------------

class TextCleaner:
# Encapsula todo el pipeline de limpieza de texto

    def __init__(self, cfg: TextCleaningConfig):
        self.cfg = cfg
        setup_nltk(cfg.language)
        self.stop_words = set(stopwords.words(cfg.language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
    #Aplica todo el proceso de limpieza a un solo texto
        # Normalización con regex (urls, menciones, minúsculas, etc.)
        text = normalize_text(text, self.cfg)
        if not text:
            return ""

        # Tokenización
        tokens = nltk.word_tokenize(text)

        # Filtros a nivel token (stopwords, longitud, etc.)
        tokens = filter_tokens(tokens, self.stop_words, self.cfg)

        # Lematización
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # Reconstruir string
        cleaned_text = " ".join(tokens)

        return cleaned_text

    def clean_series(self, series: pd.Series) -> pd.Series:
    # Aplica clean_text() a una serie de pandas
        return series.apply(self.clean_text)
        
# =========================
# PIPELINE DE DATASET
# =========================

def run_dataset_pipeline(
    input_path: Path,
    output_path: Path,
    text_column: str,
    output_text_column: str,
    cfg: TextCleaningConfig,
) -> None:
    print(f"[1/4] Leyendo dataset: {input_path}")
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(
            f"No existe la columna '{text_column}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    print(f"[2/4] Inicializando cleaner (idioma stopwords: {cfg.language})")
    cleaner = TextCleaner(cfg)

    print(f"[3/4] Limpiando columna '{text_column}' -> '{output_text_column}'")
    df[output_text_column] = cleaner.clean_series(df[text_column])

    print(f"[3b/4] Filtrando textos con menos de {cfg.min_doc_chars} caracteres")
    df = df[df[output_text_column].str.len() >= cfg.min_doc_chars].copy()

    print(f"[4/4] Guardando salida: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ Listo.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    cfg = TextCleaningConfig(
        language="english",     
        min_doc_chars=5,
        min_token_len=2,
        keep_only_alpha=True,
        to_lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_emails=True,
        remove_non_ascii=False,
    )

    run_dataset_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        text_column=TEXT_COLUMN,
        output_text_column=OUTPUT_TEXT_COLUMN,
        cfg=cfg,
    )
