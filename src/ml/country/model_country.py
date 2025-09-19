"""
Module d'entraînement d'un modèle Gradient Boosting pour la prédiction de pays.

- Colonne cible : 'country' (France, Germany, Benelux, Others)
- Utilise HistGradientBoostingClassifier avec class_weight='balanced'
- Validation croisée stratifiée conditionnelle
- Sauvegarde du modèle et de l'encodeur
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

def train_country():
    base_dir = os.path.dirname(__file__)
    path_vectors = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    path_labels = os.path.join(base_dir, "..", "..", "..", "data", "labeled.csv")
    models_dir = os.path.join(base_dir, "..", "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Chargement des données ---
    df_vectors = pd.read_csv(path_vectors, sep=";")
    df_labels = pd.read_csv(path_labels, sep=";")

    # --- Merge et nettoyage basique ---
    df = pd.merge(df_vectors, df_labels, on="doc", how="inner").dropna(subset=["country_y"])
    df = df.drop(columns=["label", "domain", "client","revenu","gotomarket","startup", "produit"], errors="ignore")

    # --- Préparer features (X) et target (y) ---
    X_df = df.drop(columns=["doc", "country_y"], errors="ignore")
    X_df = pd.DataFrame(X_df)

    if X_df.shape[1] == 0:
        raise ValueError("Aucune colonne de features trouvée (après drop doc/country).")

    X_num = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Vérifier au moins une feature informative
    if (X_num.abs().sum(axis=0) != 0).sum() == 0:
        raise ValueError("Toutes les features sont nulles — vérifie tfidf_vectors.csv.")

    # Target
    y_raw = df["country_y"].astype(str)
    if X_num.shape[0] != len(y_raw):
        raise ValueError(f"Incohérence X/y: X.shape[0]={X_num.shape[0]} vs len(y)={len(y_raw)}")

    # Encodage labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # --- Modèle Gradient Boosting ---
    model = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=42,
        class_weight='balanced'
    )

    # --- Validation croisée stratifiée conditionnelle ---
    series_counts = pd.Series(y).value_counts()
    n_min = int(series_counts.min()) if not series_counts.empty else 0
    n_splits = min(3, n_min) if n_min >= 2 else 0

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

    # --- Entraînement final sur tout le jeu ---
    model.fit(X_num, y)

    # --- Sauvegarde ---
    joblib.dump(model, os.path.join(models_dir, "country_gb_model.joblib"))
    joblib.dump(le, os.path.join(models_dir, "country_label_encoder.joblib"))
    print(f"Modèle et encodeur sauvegardés dans {models_dir}")


if __name__ == "__main__":
    train_country()
