"""
Module d'entraînement d'un modèle Gradient Boosting pour la prédiction de pays.

Ce script entraîne un modèle de classification supervisée (HistGradientBoostingClassifier)
pour prédire le pays associé à un document en fonction de ses vecteurs TF-IDF.

Fonctionnalités :
- Colonne cible : 'country' (France, Germany, Benelux, Others)
- Utilise un Gradient Boosting avec pondération automatique des classes (`class_weight='balanced'`)
- Validation croisée stratifiée conditionnelle selon la taille minimale des classes
- Sauvegarde automatique du modèle et de l'encodeur de labels dans le répertoire `models`

Fichiers utilisés :
- Données d'entrée : `data/processed/tfidf_vectors.csv`
- Étiquettes : `data/labeled.csv`
- Modèles sauvegardés : `models/country_gb_model.joblib` et `models/country_label_encoder.joblib`
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report


def train_country():
    """
    Entraîne un modèle HistGradientBoostingClassifier pour prédire le pays associé à un document.

    Étapes principales :
    1. Chargement des données TF-IDF et des labels depuis les fichiers CSV.
    2. Fusion et nettoyage des DataFrames.
    3. Préparation des features (X) et de la cible (y).
    4. Encodage des labels de pays en entiers via LabelEncoder.
    5. Entraînement d’un modèle Gradient Boosting avec pondération équilibrée des classes.
    6. Validation croisée stratifiée conditionnelle si suffisamment d’échantillons.
    7. Sauvegarde du modèle entraîné et de l’encodeur.

    Raises
    ------
    ValueError
        Si aucune colonne de features n’est trouvée ou si toutes les valeurs sont nulles.
        Si le nombre de lignes entre X et y ne correspond pas.
    """

    # --- Définition des chemins relatifs ---
    base_dir = os.path.dirname(__file__)
    path_vectors = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    path_labels = os.path.join(base_dir, "..", "..", "..", "data", "labeled.csv")
    models_dir = os.path.join(base_dir, "..", "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Chargement des données ---
    df_vectors = pd.read_csv(path_vectors, sep=";")
    df_labels = pd.read_csv(path_labels, sep=";")

    # --- Fusion sur la colonne 'doc' et suppression des lignes sans pays ---
    df = pd.merge(df_vectors, df_labels, on="doc", how="inner").dropna(subset=["country_y"])

    # Suppression de colonnes non pertinentes pour la prédiction
    df = df.drop(columns=["label", "domain", "client", "revenu", "gotomarket", "startup", "produit"], errors="ignore")

    # --- Préparation des features (X) et de la target (y) ---
    X_df = df.drop(columns=["doc", "country_y"], errors="ignore")
    X_df = pd.DataFrame(X_df)

    # Vérification qu’il reste bien des colonnes de features
    if X_df.shape[1] == 0:
        raise ValueError("Aucune colonne de features trouvée (après drop doc/country).")

    # Conversion en numérique et remplacement des NaN par 0
    X_num = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Vérification de la présence de valeurs informatives
    if (X_num.abs().sum(axis=0) != 0).sum() == 0:
        raise ValueError("Toutes les features sont nulles — vérifie tfidf_vectors.csv.")

    # --- Préparation de la variable cible ---
    y_raw = df["country_y"].astype(str)

    # Vérification de la cohérence X/y
    if X_num.shape[0] != len(y_raw):
        raise ValueError(f"Incohérence X/y: X.shape[0]={X_num.shape[0]} vs len(y)={len(y_raw)}")

    # --- Encodage des labels de pays ---
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # --- Définition du modèle Gradient Boosting ---
    model = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=42,
        class_weight='balanced'
    )

    # --- Détermination dynamique du nombre de splits pour la validation croisée ---
    series_counts = pd.Series(y).value_counts()
    n_min = int(series_counts.min()) if not series_counts.empty else 0
    n_splits = min(3, n_min) if n_min >= 2 else 0

    # --- Validation croisée stratifiée conditionnelle ---
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            y_pred = cross_val_predict(model, X_num, y, cv=cv)
            print("Classification report (cross-validation) :")
            print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))
        except Exception as e:
            print(f"cross_val_predict a échoué ({type(e).__name__}): {e}")
            print("→ On passera directement à l'entraînement final sans CV.")
    else:
        print("Cross-validation skipped: pas assez d'exemples par classe pour n_splits>=2.")

    # --- Entraînement final sur l'ensemble du jeu de données ---
    model.fit(X_num, y)

    # --- Sauvegarde du modèle et de l'encodeur ---
    joblib.dump(model, os.path.join(models_dir, "country_gb_model.joblib"))
    joblib.dump(le, os.path.join(models_dir, "country_label_encoder.joblib"))
    print(f"Modèle et encodeur sauvegardés dans {models_dir}")


# --- Point d'entrée principal ---
if __name__ == "__main__":
    train_country()
