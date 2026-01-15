# Proyecto: Predicción de depresión en comentarios de Reddit
1# Utilizar el siguiente comando para descargar los paquetes dentro del entorno local:
# pip install requirements.txt
Activar el entonro venv --> .\.venv\Scripts\Activate.ps1



Estructura básica del proyecto para empezar a trabajar en la parte técnica:
- Limpieza de datos (María)
- Preprocesado de texto (Rodrigo)
- Embeddings TF-IDF (José Luis)
- Modelo básico de clasificación (Daniel)


<<<<<<< HEAD
=======
1️⃣ limpieza_dataset.py

Responsabilidad:

Es el primer paso en el pipeline.

Se encarga de limpiar el dataset original desde su forma cruda (por ejemplo comentarios de Reddit).

Funciones típicas:

Eliminar filas vacías.

Quitar duplicados.

Normalizar texto: minúsculas, quitar URLs, markdown, emojis, caracteres no alfabéticos, espacios extra.

Filtrar comentarios cortos (< 4 palabras).

Resultado esperado:

Archivo CSV limpio (reddit_depression_clean.csv) con al menos las columnas:

clean_text → texto limpio

label → etiqueta de ansiedad/depresión, etc.

Este archivo ya no necesita limpieza estructural, es el input para los siguientes pasos.

2️⃣ cleaning_pipeline.py

Responsabilidad:

Aplica un pipeline más avanzado de limpieza y preprocesado de texto.

Funciones típicas:

Tokenización.

Eliminación de stopwords.

Lematización.

Filtrado por longitud mínima de token y texto.

Opcional: quitar números, mantener solo alfabéticos, quitar menciones, URLs.

Resultado esperado:

Archivo CSV procesado (dataset_preprocessed.csv) con columna:

processed_text → texto tokenizado, filtrado y lematizado listo para análisis.

label → etiquetas sin cambios.

Este archivo sirve para generar embeddings.

3️⃣ embeddings_tfidf.py

Responsabilidad:

Tomar el CSV procesado (dataset_preprocessed.csv) y convertir los textos a vectores TF-IDF.

Funciones típicas:

Inicializar TfidfVectorizer.

Ajustar el vectorizer sobre la columna processed_text.

Crear la matriz dispersa de features X y mantener las etiquetas y.

Guardar todo en un .pkl para uso posterior.

Resultado esperado:

Archivo .pkl (features_tfidf.pkl) con:

X → matriz TF-IDF (sparse)

y → etiquetas correspondientes

vectorizer → objeto TfidfVectorizer guardado para predecir nuevos datos

4️⃣ modelo_basico.py

Responsabilidad:

Entrenar un modelo de clasificación sobre los embeddings TF-IDF.

Funciones típicas:

Cargar features_tfidf.pkl.

Dividir datos en entrenamiento/prueba.

Entrenar modelo (por ejemplo LogisticRegression o SGDClassifier).

Evaluar el modelo (accuracy, F1, precision, recall).

Guardar modelo entrenado y vectorizer para predicciones futuras.

Resultado esperado:

Modelo entrenado (logreg_model.pkl) listo para predecir niveles de ansiedad en comentarios nuevos.

Vectorizer guardado (tfidf_vectorizer.pkl) para transformar comentarios futuros.

Métricas de evaluación impresas en consola para medir desempeño.
>>>>>>> 38140e0 (mensaje)
