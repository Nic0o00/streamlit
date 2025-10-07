# predict_domain_select.py
"""
Module de prédiction du domaine à partir de vecteurs TF-IDF.

Ce script charge un ensemble de modèles entraînés (sélection de features,
réduction de dimension, scaler, centroides et classifieur) afin de prédire
le domaine associé à chaque document vectorisé.

Étapes principales :
1. Chargement des modèles préalablement entraînés.
2. Chargement du fichier contenant les vecteurs TF-IDF.
3. Préparation et sélection des features (mêmes transformations que pour l'entraînement).
4. Calcul des similarités cosinus avec les centroides.
5. Réduction de dimension (SVD) + mise à l’échelle.
6. Prédiction du domaine via le classifieur logistique.
7. Sauvegarde des résultats dans un CSV avec les scores de confiance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --- Chemins de base ---
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
MODELS_DIR = os.path.join(BASE, "..", "..", "..", "models")
OUT = os.path.join(BASE, "..", "..", "..", "output", "predictions")
os.makedirs(OUT, exist_ok=True)
OUT_FILE = os.path.join(OUT, "tfidf_vectors_with_domain_predictions.csv")


def predict_domain():
    """
    Applique le pipeline complet de prédiction du domaine à partir des vecteurs TF-IDF.

    Étapes :
        - Charge tous les objets (sélecteur, SVD, scaler, centroides, classifieur, label encoder).
        - Recrée le même espace de features que lors de l’entraînement.
        - Calcule les similarités cosinus avec les centroides.
        - Transforme les features (SVD + standardisation).
        - Combine les features finales et exécute la prédiction du domaine.
        - Sauvegarde le résultat dans un fichier CSV avec score de confiance.

    Sorties :
        Un fichier CSV `tfidf_vectors_with_domain_predictions.csv` dans `output/predictions/`,
        contenant les colonnes :
            - doc : identifiant du document
            - predicted_domain : domaine prédit
            - confidence_score : score de confiance associé
    """

    # --- Chargement des objets enregistrés pendant l'entraînement ---
    selector = joblib.load(os.path.join(MODELS_DIR, "domain_selector.joblib"))
    svd = joblib.load(os.path.join(MODELS_DIR, "domain_svd.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "domain_scaler.joblib"))
    centroids = joblib.load(os.path.join(MODELS_DIR, "domain_centroids.joblib"))
    clf = joblib.load(os.path.join(MODELS_DIR, "domain_clf.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "domain_label_encoder.joblib"))
    nonzero_cols = joblib.load(os.path.join(MODELS_DIR, "domain_nonzero_columns.joblib"))

    # --- Chargement des vecteurs TF-IDF à prédire ---
    df_vec = pd.read_csv(VECT_CSV, sep=";")
    df_vec["doc"] = df_vec["doc"].astype(str)
    docs = df_vec["doc"].tolist()

    # --- Vérification / ajout des colonnes manquantes (cohérence features) ---
    for c in nonzero_cols:
        if c not in df_vec.columns:
            df_vec[c] = 0

    # --- Préparation des features ---
    X_df = df_vec[nonzero_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_sp = csr_matrix(X_df.values)

    # --- Sélection des features (chi2) ---
    X_sel = selector.transform(X_sp)  # matrice sparse

    # --- Calcul des similarités cosinus avec les centroides des classes ---
    cos_sim = cosine_similarity(X_sel, centroids)  # matrice dense

    # --- Réduction de dimension (SVD) + mise à l’échelle ---
    X_red = svd.transform(X_sel)
    X_scaled = scaler.transform(X_red)

    # --- Combinaison des features réduites + similarités cosinus ---
    X_final = np.hstack([X_scaled, cos_sim])

    # --- Prédiction du domaine + score de confiance ---
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_final)
        pred_idx = np.argmax(proba, axis=1)
        maxp = proba.max(axis=1)
    else:
        pred_idx = clf.predict(X_final)
        maxp = np.ones(len(pred_idx))

    # --- Conversion des indices en labels d’origine ---
    pred_lab = le.inverse_transform(pred_idx)

    # --- Construction du DataFrame de sortie ---
    out = pd.DataFrame({
        "doc": docs,
        "predicted_domain": pred_lab,
        "confidence_score": maxp
    })

    # --- Sauvegarde des résultats ---
    out.to_csv(OUT_FILE, sep=";", index=False)
    print("Predictions saved to", OUT_FILE)


if __name__ == "__main__":
    # Exécution directe du script pour lancer la prédiction
    predict_domain()
