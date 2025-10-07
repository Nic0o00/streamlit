"""
Module : predict_tech
=====================

Ce module réalise la **prédiction multi-label** des technologies d’un document
(`hard`, `soft`, ou `both`) à partir de vecteurs TF-IDF et d’un modèle
de **régression logistique One-vs-Rest** préalablement entraîné.

Fonctionnalités principales :
-----------------------------
- Chargement du modèle de classification multi-label (`lr_multilabel_techno_model.joblib`)
- Chargement des vecteurs TF-IDF depuis `data/processed/tfidf_vectors.csv`
- Calcul des probabilités pour chaque label (`hard`, `soft`)
- Application d’un seuil de confiance (`0.5` par défaut)
- Fusion des résultats pour produire un label combiné :  
  `"hard"`, `"soft"`, `"both"`, ou `"unknown"`
- Export des résultats dans un CSV contenant :  
  `doc`, `predicted_tech`, et `confidence_score`

Fichier de sortie :
-------------------
- `output/predictions/tfidf_vectors_with_tech_predictions.csv`

Auteur :
--------
Fait partie du pipeline de prédiction documentaire multi-label.
"""

import os
import pandas as pd
import joblib
import numpy as np


def predict_tech():
    """
    Effectue les prédictions de labels 'hard' et 'soft' sur un jeu de vecteurs TF-IDF.

    Description détaillée
    ----------------------
    1. Charge le modèle de régression logistique multi-label (One-vs-Rest)
    2. Charge les vecteurs TF-IDF et vérifie que toutes les features du modèle sont présentes
    3. Calcule les probabilités de chaque label pour chaque document
    4. Applique un seuil de confiance (0.5) pour déterminer la présence du label
    5. Fusionne les deux sorties (`hard` / `soft`) en un label combiné (`both`, `hard`, `soft`, `unknown`)
    6. Sauvegarde les résultats avec le score de confiance global

    Notes
    -----
    - Le modèle doit avoir été entraîné et sauvegardé au préalable via `train_tech.py`.
    - Le seuil de prédiction est configurable via la variable `threshold`.
    - Si une feature attendue par le modèle est absente du CSV, elle est ajoutée et remplie par des zéros.
    - La confiance globale est le **maximum** entre la proba 'hard' et la proba 'soft'.

    Returns
    -------
    None
        Les prédictions sont sauvegardées dans un fichier CSV à la fin de l’exécution.
    """

    # --- Seuil de probabilité pour considérer un label comme présent ---
    threshold = 0.5

    # --- Définition des chemins relatifs ---
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "..", "..", "models", "lr_multilabel_techno_model.joblib")
    vectors_path = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")

    output_dir = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tfidf_vectors_with_tech_predictions.csv")

    # --- Chargement du modèle entraîné ---
    clf = joblib.load(model_path)

    # --- Chargement des vecteurs TF-IDF complets ---
    df_vectors_full = pd.read_csv(vectors_path, sep=";")

    # --- Vérification que toutes les features attendues par le modèle existent ---
    feature_names = clf.estimators_[0].feature_names_in_
    missing_feats = [feat for feat in feature_names if feat not in df_vectors_full.columns]

    # Si certaines features manquent, les ajouter avec valeur 0 (aucune occurrence)
    if missing_feats:
        zeros_df = pd.DataFrame(0, index=df_vectors_full.index, columns=missing_feats)
        df_vectors_full = pd.concat([df_vectors_full, zeros_df], axis=1)

    # --- Sélection des colonnes correspondant aux features du modèle ---
    X = df_vectors_full[feature_names]

    # --- Prédictions de probabilité pour chaque classe (hard / soft) ---
    hard_proba_all = clf.estimators_[0].predict_proba(X)
    soft_proba_all = clf.estimators_[1].predict_proba(X)

    # --- Gestion des cas où une seule classe est présente (1D) ---
    if hard_proba_all.shape[1] == 1:
        hard_probs = np.zeros(len(X))
    else:
        hard_probs = hard_proba_all[:, 1]

    if soft_proba_all.shape[1] == 1:
        soft_probs = np.zeros(len(X))
    else:
        soft_probs = soft_proba_all[:, 1]

    # --- Calcul du score de confiance global ---
    global_confidence = np.maximum(hard_probs, soft_probs)

    # --- Application du seuil de décision ---
    hard_pred = np.where(hard_probs >= threshold, "hard", "unknown")
    soft_pred = np.where(soft_probs >= threshold, "soft", "unknown")

    # --- Fusion des deux prédictions en un seul label final ---
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

    # --- Construction du DataFrame final de résultats ---
    df_results = pd.DataFrame({
        "doc": df_vectors_full["doc"],
        "predicted_tech": combined_pred,
        "confidence_score": global_confidence
    })

    # --- Sauvegarde des résultats ---
    df_results.to_csv(output_file, sep=";", index=False)
    print(f"✅ Prédictions sauvegardées dans : {output_file}")


# --- Point d’entrée principal ---
if __name__ == "__main__":
    predict_tech()
