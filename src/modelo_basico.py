import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_PATH = Path("data/features_tfidf.pkl")

def main():
    print(f"Cargando features desde {FEATURES_PATH}...")
    with open(FEATURES_PATH, "rb") as f:
        X, y, vectorizer = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== RESULTADOS DEL MODELO (Regresión Logística) ===")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar modelo entrenado
    model_path = Path("models/model_logreg_tfidf.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()
