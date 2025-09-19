import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np


def train_gtm_model(
    tfidf_path=None,
    labeled_path=None,
    models_dir=None,
    test_size=0.3,
    random_state=42
):
    base_dir = os.path.dirname(__file__)
    tfidf_path = tfidf_path or os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    labeled_path = labeled_path or os.path.join(base_dir, "..", "..", "..", "data", "labeled.csv")
    models_dir = models_dir or os.path.join(base_dir, "..", "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Chargement des données avec encodage robuste
    df_vec = pd.read_csv(tfidf_path, sep=";", encoding="utf-8-sig")
    df_lab = pd.read_csv(labeled_path, sep=";", encoding="utf-8-sig")

    # Merge sur 'doc'
    df = pd.merge(df_vec, df_lab[["doc", "gotomarket"]], on="doc", how="inner")
    if df.shape[0] == 0:
        raise RuntimeError("Aucune ligne après merge (vérifie 'doc').")
    df = df.dropna(subset=["gotomarket"])

    # Préparer X et y
    X = df.drop(columns=["doc", "gotomarket"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["gotomarket"].fillna(0).astype(int)
    print(y.value_counts())

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Définir les modèles à tester
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state, class_weight="balanced"),
        "LogisticRegression": LogisticRegression(max_iter=500, solver="saga", n_jobs=-1, class_weight="balanced"),
        "LinearSVC": LinearSVC(max_iter=2000, class_weight="balanced")
    }

    best_model = None
    best_score = -np.inf
    best_name = None

    results = {}

    # Entraînement et évaluation
    for name, model in models.items():
        print(f"\n--- Entraînement {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="macro")
        results[name] = score
        print(classification_report(y_test, y_pred))
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # Sauvegarde du meilleur modèle et des features
    joblib.dump(best_model, os.path.join(models_dir, "gtm_model.joblib"))
    joblib.dump(list(X.columns), os.path.join(models_dir, "gtm_feature_names.joblib"))

    # Sauvegarde des résultats
    results_path = os.path.join(models_dir, "gtm_model_metrics.json")
    import json
    with open(results_path, "w") as f:
        json.dump({"results": results, "best_model": best_name, "best_score": best_score}, f, indent=4)

    print("\nEntraînement GTM terminé.")
    print(f"Meilleur modèle: {best_name} (F1-macro={best_score:.3f})")
    print(f"Modèle sauvegardé: {os.path.join(models_dir, 'gtm_model.joblib')}")
    print(f"Métriques sauvegardées: {results_path}")


if __name__ == "__main__":
    train_gtm_model()
