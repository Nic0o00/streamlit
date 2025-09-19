"""
Module de prédiction multi-label pour la technologie des documents.

Ce module charge un modèle Logistic Regression entraîné pour prédire les labels
'hard' et 'soft' à partir des vecteurs TF-IDF, calcule les probabilités, applique
un seuil de confiance, fusionne les labels en un label combiné, et sauvegarde
les résultats dans un fichier CSV.

Fonctions
---------
predict_tech()
    Charge le modèle et les vecteurs, effectue les prédictions multi-label,
    et sauvegarde les résultats avec scores de confiance.
"""
import os
import pandas as pd
import joblib
import numpy as np

def predict_tech():
    """
    Effectue les prédictions de labels 'hard' et 'soft' sur un jeu de vecteurs TF-IDF.

    Notes
    -----
    - Seuil de prédiction par défaut : 0.5.
    - Les probabilités de chaque label sont calculées pour chaque document.
    - Fusion des labels en un label unique : 'hard', 'soft', 'both', ou 'unknown'.
    - La confiance globale est définie comme le maximum des probabilités 'hard' et 'soft'.
    - Les résultats sont sauvegardés dans `output/predictions/tfidf_vectors_with_tech_predictions.csv`.

    Returns
    -------
    None
    """
    threshold=0.5

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "..", "..", "models", "lr_multilabel_techno_model.joblib")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")

    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tfidf_vectors_with_tech_predictions.csv")

    # Charger modèle
    clf = joblib.load(model_path)

    # Charger vecteurs
    df_vectors_full = pd.read_csv(vectors_path, sep=";")

    # Vérifier features
    feature_names = clf.estimators_[0].feature_names_in_
    # Ajouter toutes les colonnes manquantes en une seule fois pour éviter la fragmentation
    missing_feats = [feat for feat in feature_names if feat not in df_vectors_full.columns]
    if missing_feats:
        zeros_df = pd.DataFrame(0, index=df_vectors_full.index, columns=missing_feats)
        df_vectors_full = pd.concat([df_vectors_full, zeros_df], axis=1)

    X = df_vectors_full[feature_names]

    # Prédictions proba pour chaque sortie
    hard_proba_all = clf.estimators_[0].predict_proba(X)
    soft_proba_all = clf.estimators_[1].predict_proba(X)

    # Gestion des cas 1D
    if hard_proba_all.shape[1] == 1:
        hard_probs = np.zeros(len(X))
    else:
        hard_probs = hard_proba_all[:, 1]

    if soft_proba_all.shape[1] == 1:
        soft_probs = np.zeros(len(X))
    else:
        soft_probs = soft_proba_all[:, 1]

    # Confiance globale = max des deux
    global_confidence = np.maximum(hard_probs, soft_probs)

    # Application du seuil
    hard_pred = np.where(hard_probs >= threshold, "hard", "unknown")
    soft_pred = np.where(soft_probs >= threshold, "soft", "unknown")

    # Fusion en un seul label
    combined_pred = []
    for h, s in zip(hard_pred, soft_pred):
        if h == "hard" and s == "soft":
            combined_pred.append("both")
        elif h == "hard":
            combined_pred.append("hard")
        elif s == "soft":
            combined_pred.append("soft")
        else:
            combined_pred.append("unknown")

    # Résultats finaux
    df_results = pd.DataFrame({
        "doc": df_vectors_full["doc"],
        "predicted_tech": combined_pred,
        "confidence_score": global_confidence
    })

    # Sauvegarde
    df_results.to_csv(output_file, sep=";", index=False)
    print(f"Prédictions sauvegardées dans : {output_file}")

if __name__ == "__main__":
    predict_tech()
