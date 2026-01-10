

from __future__ import annotations # Permite usar list[str] sin problemas de versión

import re                          # Expresiones regulares
from dataclasses import dataclass  # Para crear clases de configuración simples
from pathlib import Path           # Manejo moderno de rutas
from typing import Iterable        # Tipado para colecciones

import pandas as pd                # Manejo de datasets preventing
import nltk                        # NLP básico
from nltk.corpus import stopwords  # Stopwords
from nltk.stem import WordNetLemmatizer  # Lematizador

#------------------------------------------------
#CONFIGURACIÓN
#------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "reddit_depression_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "dataset_preprocessed.csv"

TEXT_COLUMN = "clean_text" # Columna de texto de entrada
OUTPUT_TEXT_COLUMN = "processed_text" # Columna de texto de salida


# Config global para el pipeline de limpieza.

@dataclass
class TextCleaningConfig:
   
    language: str = "english"   # "english", "spanish", etc. (según NLTK)
    min_doc_length: int = 5     # Longitud mínima del texto final (caracteres)
    min_token_length: int = 2   # Longitud mínima de cada token
    keep_only_alpha: bool = True  # Mantener solo tokens alfabéticos
    remove_numbers: bool = True # Eliminar tokens numéricos
    remove_urls: bool = True # URLs
    remove_mentions: bool = True  # @usuario
    to_lowercase: bool = True # Minúsculas

#------------------------------------------------
#NTLK SETUP
#------------------------------------------------

def setup_nltk(language: str = "english") -> None: # Descarga recursos necesarios de NLTK si no están
    nltk.download("punkt", quiet=True) 
    nltk.download("punkt_tab", quiet=True)      # Tokenizador
    nltk.download("stopwords", quiet=True)   # Stopwords
    nltk.download("wordnet", quiet=True)     # WordNet (lematización)
# Si cambiamos a otro idioma, NLTK tiene qye tener las stopwords en ese idioma
    stopwords.words(language)  # fuerza la carga de stopwords del idioma

#------------------------------------------------
#FUNCIONES DE LIMPIEZA
#------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")         # URLs
MENTION_PATTERN = re.compile(r"@\w+")                      # @usuario
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")  # emails
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7f]")            # emojis / unicode raro



def normalize_text(text: str, cfg: TextCleaningConfig) -> str:
#Aplica limpieza básica antes del tokenizado (minisculas, URLs, menciones, etc.)
    if not isinstance(text, str): # Si no es string (NaN, None, número) → texto vacío
        return ""

    # Convierte aminúsculas
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
    print("[1/4] Leyendo dataset: Completado")

    print(f"[2/4] Inicializando cleaner (idioma stopwords: {cfg.language})")
    cleaner = TextCleaner(cfg)
    print("[2/4] Inicializando cleaner: Completado")

    print(f"[3/4] Limpiando columna '{text_column}' -> '{output_text_column}'")
    df[output_text_column] = cleaner.clean_series(df[text_column])
    print("[3/4] Limpiando columna")

    print(f"[3b/4] Filtrando textos con menos de {cfg.min_doc_length} caracteres")
    df = df[df[output_text_column].str.len() >= cfg.min_doc_length].copy()
    print("[3b/4] Filtrando textos con menos de n caracteres")

    print(f"[4/4] Guardando salida: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Listo.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    # Configuración del pipeline
    cfg = TextCleaningConfig(
    language="english",
    min_doc_length=5,
    min_token_length=2,
    keep_only_alpha=True,
    remove_numbers=True,
    remove_urls=True,
    remove_mentions=True,
    to_lowercase=True,
)
# Ejecutar pipeline
    run_dataset_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        text_column=TEXT_COLUMN,
        output_text_column=OUTPUT_TEXT_COLUMN,
        cfg=cfg,
    )
