import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Chemins ---
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier_rf.joblib")

def main():
    # --- Charger vecteurs et labels ---
    X = pd.read_csv(VECT_CSV, sep=";")  # vecteurs TF-IDF avec colonne 'doc'
    df_labels = pd.read_csv(LABELED_CSV, sep=";", encoding="ISO-8859-1")

    # --- Nettoyer les labels : garder uniquement les valeurs autorisées ---
    allowed_labels = ["Interessant", "Unfavorable", "Very Unfavorable", "Out"]
    df_labels = df_labels[df_labels["resultat"].isin(allowed_labels)]

    # --- Filtrer X pour ne garder que les vecteurs ayant un label ---
    X_train_vectors = X[X["doc"].isin(df_labels["doc"])].reset_index(drop=True)

    # --- Récupérer les labels correspondants ---
    y_train_labels = df_labels[df_labels["doc"].isin(X_train_vectors["doc"])].reset_index(drop=True)["resultat"]

    # --- Supprimer la colonne 'doc' pour l'entraînement ---
    X_train_vectors = X_train_vectors.drop(columns=["doc"])

    # --- Entraîner le modèle RandomForest sur **toutes les données** ---
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_vectors, y_train_labels)

    # --- Optionnel : évaluation sur les mêmes données d'entraînement ---
    y_pred = clf.predict(X_train_vectors)
    print("=== Évaluation sur les données d'entraînement ===")
    print(classification_report(y_train_labels, y_pred))

    # --- Sauvegarder le modèle ---
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Modèle RandomForest sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    main()
