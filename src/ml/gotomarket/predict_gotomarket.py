"""
Prédiction de l'année Go-To-Market pour des documents à partir d'un modèle entraîné.
Sortie CSV : doc, gotomarket
"""

import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import classification_report

def predict_gtm():
    base_dir = os.path.dirname(__file__)

    models_dir = os.path.join(base_dir, "..", "..", "..", "models")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)

   
    output_file = os.path.join(output_dir, "gtm_predictions_.csv")

    # Charger vecteurs avec encodage robuste
    df_vec = pd.read_csv(vectors_path, sep=";", encoding="utf-8-sig")
    docs = df_vec["doc"].tolist() if "doc" in df_vec.columns else list(range(len(df_vec)))
    X = df_vec.drop(columns=["doc"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)

    # Charger modèle + features
    clf = joblib.load(os.path.join(models_dir, "gtm_model.joblib"))
    feature_names = joblib.load(os.path.join(models_dir, "gtm_feature_names.joblib"))

    # Aligner colonnes : ajouter les manquantes, supprimer les en trop
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]

    # Prédictions
    y_pred = clf.predict(X)

    df_out = pd.DataFrame({
        "doc": docs,
        "gotomarket": y_pred
    })
    df_out.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig")

    print(f"✅ Prédictions GTM sauvegardées dans : {output_file}")

    # Si les labels réels existent dans le CSV, calculer un rapport
    if "gotomarket" in df_vec.columns:
        y_true = df_vec["gotomarket"].fillna(0).astype(int)
        print("\nRapport de classification (sur données annotées) :")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    predict_gtm()