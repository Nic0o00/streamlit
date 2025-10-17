"""
Module de prédiction `predict_resultat`.

Ce script charge un modèle RandomForest préalablement entraîné sur des vecteurs TF-IDF 
et prédit le label `resultat` (classe du document) pour chaque document disponible 
dans le fichier de vecteurs.

Fonction principale :
    - `predict_resultat()`: applique le modèle sur tous les documents pour générer des prédictions
      et sauvegarde les résultats dans un fichier CSV avec un score de confiance.

Entrées :
    - models/deck_classifier_rf.joblib : modèle entraîné (RandomForest)
    - data/processed/tfidf_vectors.csv : vecteurs TF-IDF avec identifiant 'doc'

Sortie :
    - output/predictions/tfidf_vectors_with_resultat_predictions.csv : fichier CSV contenant :
        * doc : identifiant du document
        * predicted_resultat : classe prédite
        * confidence_score : probabilité maximale associée à la prédiction
"""

import os
import joblib
import pandas as pd

# --- Définition des chemins de base ---
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier_rf.joblib")
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
OUTPUT_CSV = os.path.join(BASE, "..", "..", "..", "output", "predictions", "tfidf_vectors_with_resultat_predictions.csv")

def predict_resultat():
    """
    Applique le modèle RandomForest pour prédire la classe (`resultat`)
    de chaque document à partir de ses vecteurs TF-IDF.
    """
    # --- Charger le modèle RandomForest sauvegardé ---
    clf = joblib.load(MODEL_PATH)
    
    # --- Charger les vecteurs TF-IDF ---
    X = pd.read_csv(VECT_CSV, sep=";")

    # --- Nettoyage ---
    X = X.dropna(subset=["doc"]).reset_index(drop=True)

    # --- Préparer les noms de documents ---
    doc_names = X["doc"]

    # --- Préparer les features pour la prédiction ---
    X_vectors = X.drop(columns=["doc"])

    # --- Réindexer pour correspondre aux colonnes utilisées à l'entraînement ---
    train_columns = clf.feature_names_in_
    X_vectors = X_vectors.reindex(columns=train_columns, fill_value=0)

    # --- Prédictions ---
    preds = clf.predict(X_vectors)
    probs = clf.predict_proba(X_vectors).max(axis=1)

    # --- Créer le DataFrame résultat ---
    results = pd.DataFrame({
        "doc": doc_names,
        "predicted_resultat": preds,
        "confidence_score": probs
    })

    # --- Sauvegarder les résultats ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="ISO-8859-1")
    print(f"Résultats sauvegardés dans {OUTPUT_CSV}")

# --- Point d’entrée du script ---
if __name__ == "__main__":
    predict_resultat()
