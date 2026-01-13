

rom __future__ import annotations

# LIBRERIAS UTILIZADAS

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import nltk # #Librería de procesamiento de lenguaje natural
from nltk.corpus import stopwords #Lista de palabras vacías (stopwords) para diferentes idiomas

from nltk.stem import WordNetLemmatizer #Lemanizador para convertir palabras a su forma base

from nltk.tokenize import wordpunct_tokenize #Tokenizador que divide el texto en palabras y signos de puntuación

from tqdm import tqdm #Librería para mostrar barras de progreso en terminal

tqdm.pandas()


# RUTAS

BASE_DIR = Path(__file__).resolve().parent # carpeta base del proyecto

INPUT_PATH = BASE_DIR / "data" / "reddit_depression_clean.csv" # Ruta del archivo de entrada (clean)

OUTPUT_CSV_PATH = BASE_DIR / "data" / "dataset_preprocess.csv" # Ruta del archivo CSV de salida

TEXT_COLUMN = "clean_text"
OUTPUT_TEXT_COLUMN = "processed_text"


# CONFIGURACION DE LIMPIEZA DE TEXTO

@dataclass
class TextCleaningConfig:
    language: str = "english"
    min_doc_length: int = 5
    min_token_length: int = 2
    keep_only_alpha: bool = True
    remove_numbers: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    to_lowercase: bool = True
    remove_non_ascii: bool = True

# AJUSTES NLTK

def setup_nltk(language: str) -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _ = stopwords.words(language)

# REGEX (PATRONES)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7f]")


# FUNCIONES DE LIMPIEZA

def normalize_text(text: str, cfg: TextCleaningConfig) -> str:
    if not isinstance(text, str):
        return ""

    if cfg.to_lowercase:
        text = text.lower()

    if cfg.remove_urls:
        text = URL_PATTERN.sub(" ", text)

    if cfg.remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)

    text = EMAIL_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    if cfg.remove_non_ascii:
        text = NON_ASCII_PATTERN.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_tokens(tokens: Iterable[str], stop_words: set[str], cfg: TextCleaningConfig) -> list[str]:
    cleaned: list[str] = []
    for t in tokens:
        if cfg.keep_only_alpha and not t.isalpha():
            continue
        if cfg.remove_numbers and any(c.isdigit() for c in t):
            continue
        if t in stop_words:
            continue
        if len(t) < cfg.min_token_length:
            continue
        cleaned.append(t)
    return cleaned


class TextCleaner:
    def __init__(self, cfg: TextCleaningConfig):
        self.cfg = cfg
        setup_nltk(cfg.language)
        self.stop_words = set(stopwords.words(cfg.language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = normalize_text(text, self.cfg)
        if not text:
            return ""
        tokens = wordpunct_tokenize(text)
        tokens = filter_tokens(tokens, self.stop_words, self.cfg)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

# PIPELINE

def run_dataset_pipeline() -> None:
    print(f"Leyendo dataset: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False) # Lee nuestro dataset clean
    print("Lectura Completada")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"No existe la columna '{TEXT_COLUMN}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )


    cfg = TextCleaningConfig()
    print("Inicializando cleaner") # Inicializa el limpiador de texto con la configuración realizada
    cleaner = TextCleaner(cfg)

    print(f"Procesando -> {OUTPUT_TEXT_COLUMN}") # Aplica la limpieza de texto a cada fila de la columna especificada
    df[OUTPUT_TEXT_COLUMN] = ( 
        df[TEXT_COLUMN].fillna("").astype(str).progress_apply(cleaner.clean_text)
    )

    print(f"Filtrando textos < {cfg.min_doc_length} caracteres") # Filtra los textos que son más cortos que la longitud mínima especificada
    df = df[df[OUTPUT_TEXT_COLUMN].str.len().fillna(0) >= cfg.min_doc_length].copy()
    print("Completado filtrado textos")

    
    df_out = df[[OUTPUT_TEXT_COLUMN]].copy() # Crea un nuevo DataFrame con solo la columna procesada


    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print("Guardado OK en:", OUTPUT_CSV_PATH)
    print("Proceso completado")

    # MAIN

if __name__ == "__main__":

    run_dataset_pipeline()
