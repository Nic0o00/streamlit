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

Auteur :
    Ce module s’inscrit dans la pipeline de classification documentaire par vecteurs TF-IDF.
"""

import os
import joblib
import pandas as pd

# --- Définition des chemins de base ---
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier_rf.joblib")
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")  # (non utilisé ici, mais gardé pour cohérence du pipeline)
OUTPUT_CSV = os.path.join(BASE, "..", "..", "..", "output", "predictions", "tfidf_vectors_with_resultat_predictions.csv")

def predict_resultat():
    """
    Applique le modèle RandomForest pour prédire la classe (`resultat`)
    de chaque document à partir de ses vecteurs TF-IDF.

    Étapes :
        1. Chargement du modèle entraîné depuis le répertoire `models/`.
        2. Chargement des vecteurs TF-IDF à prédire.
        3. Nettoyage des données (suppression des lignes sans 'doc').
        4. Application du modèle pour générer la prédiction et la probabilité associée.
        5. Export des résultats dans un fichier CSV de sortie.

    Sortie :
        Fichier CSV contenant les prédictions et le score de confiance pour chaque document.
    """
    # --- Charger le modèle RandomForest sauvegardé ---
    clf = joblib.load(MODEL_PATH)
    
    # --- Charger les vecteurs TF-IDF ---
    # Fichier contenant la colonne 'doc' + features du document
    X = pd.read_csv(VECT_CSV, sep=";")

    # --- Nettoyage ---
    # Suppression des lignes où 'doc' est manquant (sécurité)
    X = X.dropna(subset=["doc"]).reset_index(drop=True)

    # --- Préparation des noms de documents ---
    # Conservation des identifiants pour les associer aux prédictions
    doc_names = X["doc"]

    # --- Préparation des features pour la prédiction ---
    # Suppression de la colonne 'doc' avant passage au modèle
    X_vectors = X.drop(columns=["doc"])

    # --- Prédictions ---
    # Prédiction de la classe (label)
    preds = clf.predict(X_vectors)
    # Score de confiance = probabilité maximale parmi toutes les classes
    probs = clf.predict_proba(X_vectors).max(axis=1)

    # --- Création du DataFrame résultat ---
    # Contient uniquement les informations essentielles pour la sortie
    results = pd.DataFrame({
        "doc": doc_names,
        "predicted_resultat": preds,
        "confidence_score": probs
    })

    # --- Sauvegarde du fichier de résultats ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="ISO-8859-1")
    print(f"Résultats sauvegardés dans {OUTPUT_CSV}")

# --- Point d’entrée du script ---
if __name__ == "__main__":
    predict_resultat()
