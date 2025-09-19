"""
Module d'entraînement d'un classifieur multi-label pour la technologie des documents
avec TF-IDF enrichi (1-3 grams).
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

def train_tech():
    """
    Entraîne un modèle Logistic Regression multi-label pour les labels 'hard' et 'soft'
    à partir du TF-IDF enrichi.
    """
    # --- Chemins ---
    base_dir = os.path.dirname(__file__)
    path_vectors = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
    path_labels  = os.path.join(base_dir, "..", "..", "..", "data", "labeled.csv")
    models_dir   = os.path.join(base_dir, "..","..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Lecture des données ---
    df_vec = pd.read_csv(path_vectors, sep=";")
    df_lab = pd.read_csv(path_labels, sep=";")

    # --- Nettoyage et type ---
    df_vec["doc"] = df_vec["doc"].astype(str)
    df_lab["doc"] = df_lab["doc"].astype(str)
    df_lab = df_lab[["doc", "tech"]]  # on ne garde que les colonnes nécessaires
   

    # --- Harmonisation des noms de doc ---
    df_vec["doc"] = df_vec["doc"].str.replace(r"\.txt$", "", regex=True)
   
    # --- Merge ---
    df = pd.merge(df_vec, df_lab, on="doc", how="inner")


    print(f"Lignes après merge : {len(df)}")
    if len(df) == 0:
        raise ValueError("Aucune correspondance entre vos vecteurs et vos labels. Vérifiez les noms de doc.")

    # --- Nettoyage colonnes inutiles ---
    drop_cols = ["domain", "country", "client_y","revenu","gotomarket","startup", "produit"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- Supprimer lignes sans label ---
    df = df.dropna(subset=["tech_y"])
    if len(df) == 0:
        raise ValueError("Aucune ligne avec label valide après nettoyage.")

    # --- Colonnes binaires ---
    df["hard"] = df["tech_y"].apply(lambda x: 1 if x in ["hard", "both"] else 0)
    df["soft"] = df["tech_y"].apply(lambda x: 1 if x in ["soft", "both"] else 0)

    # --- Features et target ---
    X = df.drop(columns=["doc", "tech_y", "hard", "soft"])
    y = df[["hard", "soft"]]

    print(f"Nombre de lignes après nettoyage : {len(df)}")
    print(df[["doc", "tech_y", "hard", "soft"]].head())

    # --- Stratification ---
    stratify_labels = df["hard"].astype(str) + df["soft"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify_labels
    )

    # --- Modèle Logistic Regression One-vs-Rest ---
    base_clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=2.0,
        max_iter=2000,
        class_weight="balanced"
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)

    # --- Entraînement ---
    clf.fit(X_train, y_train)

    # --- Évaluation ---
    y_pred = clf.predict(X_test)
    print("Rapport classification pour 'hard' :")
    print(classification_report(y_test["hard"], y_pred[:, 0], zero_division=0))
    print("Rapport classification pour 'soft' :")
    print(classification_report(y_test["soft"], y_pred[:, 1], zero_division=0))

    # --- Entraînement final sur tout le dataset ---
    clf.fit(X, y)

    # --- Sauvegarde du modèle ---
    model_path = os.path.join(models_dir, "lr_multilabel_techno_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    train_tech()
