import pandas as pd
import re
import os

# Subir el dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "1_dataset_preprocessed.csv"
)

df = pd.read_csv(
    DATA_PATH,
    dtype=str,
    low_memory=False
)

print("Dataset cargado")

# Eliminar filas con texto nulo
df = df.dropna(subset=["body"])
print("Filas con texto nulo eliminadas")

# Eliminar duplicados
df = df.drop_duplicates(subset=["body"])
print("Filas duplicadas eliminadas")

# Normalización del texto
def basic_cleaning(text):
    # Pasar a minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Eliminar markdown de Reddit
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # Eliminar saltos de línea
    text = re.sub(r"\n", " ", text)

    # Eliminar caracteres no alfabéticos
    text = re.sub(r"[^a-z\s]", "", text)

    # Eliminar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["clean_text"] = df["body"].apply(basic_cleaning)

print("Limpieza estructural completada")

# Eliminar los emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # transporte y mapas
        "\U0001F1E0-\U0001F1FF"  # banderas
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

df["clean_text"] = df["clean_text"].apply(remove_emojis)

print("Emojis eliminados")

# Eliminar comentarios de menos de 4 palabras
MIN_WORDS = 4
df = df[df["clean_text"].apply(lambda x: len(x.split()) >= MIN_WORDS)]
print("comentarios de menos de 4 palabras eliminados")

# Crear el archivo final con el texto limpio
df_final = df[["clean_text", "label"]]
df_final.to_csv("reddit_depression_clean.csv", index=False)

print("Limpieza completada!!!")