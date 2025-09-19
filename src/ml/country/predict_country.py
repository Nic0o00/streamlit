"""
Module `predict_country` pour prédire le pays d'un document.

- Utilise le modèle Gradient Boosting entraîné sur vecteurs TF-IDF
- Applique un seuil pour filtrer les prédictions incertaines
- Sauvegarde les résultats avec toutes les probabilités
"""

import os
import pandas as pd
import joblib
import numpy as np

def predict_country():
    """
    Prédit le pays à partir de vecteurs TF-IDF.

    Paramètres:
    ----------
    threshold : float
        Seuil minimal de probabilité pour accepter une prédiction.
        Si probabilité maximale < threshold, pays marqué "unknown".
    """
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "..", "..", "models", "country_gb_model.joblib")
    le_path = os.path.join(base_dir, "..", "..", "..", "models", "country_label_encoder.joblib")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")

    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tfidf_vectors_with_country_predictions.csv")
    
    threshold=0.3

    # --- Chargement du modèle Gradient Boosting et encodeur ---
    model = joblib.load(model_path)
    le = joblib.load(le_path)

    # --- Chargement des vecteurs TF-IDF ---
    df_vectors_full = pd.read_csv(vectors_path, sep=";")

    # --- Préparation des features ---
    X = df_vectors_full.drop(columns=["doc"], errors="ignore")

    # Vérifier et compléter les colonnes manquantes attendues par le modèle
    feature_names = model.feature_names_in_
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0

    # Réordonner les colonnes exactement comme le modèle
    X = X[feature_names]

    # --- Prédiction des probabilités ---
    proba_all = model.predict_proba(X)

    best_proba = np.max(proba_all, axis=1)
    best_class = np.argmax(proba_all, axis=1)

    # --- Application du seuil ---
    predictions = [
        le.inverse_transform([cls_idx])[0] if prob >= threshold else "unknown"
        for prob, cls_idx in zip(best_proba, best_class)
    ]

    # --- Création du DataFrame de sortie ---
    df_results = df_vectors_full[["doc"]].copy() if "doc" in df_vectors_full.columns else pd.DataFrame({"doc": range(len(predictions))})
    df_results["predicted_country"] = predictions
    df_results["confidence"] = best_proba

    # Ajouter toutes les probabilités pour debug
    for idx, class_name in enumerate(le.classes_):
        df_results[f"proba_{class_name}"] = proba_all[:, idx]

    # --- Sauvegarde ---
    df_results.to_csv(output_file, sep=";", index=False)
    print(f"Prédictions pays sauvegardées avec debug dans : {output_file}")


if __name__ == "__main__":
    predict_country()
