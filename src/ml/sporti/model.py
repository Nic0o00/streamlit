import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# chemins
BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier.joblib")

# labels valides
VALID_LABELS = ["Out", "Very Unfavorable", "Unfavorable", "Favorable", "Very Favorable"]

def main():
    # charger vecteurs et labels
    X = pd.read_csv(VECT_CSV, sep=";")  # vecteurs TF-IDF avec colonne 'doc'
    df_labels = pd.read_csv(LABELED_CSV, sep=";", encoding="ISO-8859-1")

    # filtrer X pour ne garder que les vecteurs ayant un label
    X = X[X["doc"].isin(df_labels["doc"])].reset_index(drop=True)
    df_labels = df_labels[df_labels["doc"].isin(X["doc"])].reset_index(drop=True)

    # garder uniquement les lignes avec un label valide
    mask = df_labels["resultat"].isin(VALID_LABELS)
    X_train_vectors = X[mask].reset_index(drop=True)
    y_train_labels = df_labels.loc[mask, "resultat"].reset_index(drop=True)

    # supprimer la colonne 'doc' pour l'entraînement
    X_train_vectors = X_train_vectors.drop(columns=["doc"])

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_vectors, y_train_labels, test_size=0.2, random_state=42, stratify=y_train_labels
    )

    # modèle
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train, y_train)

    # évaluation
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # sauvegarde modèle
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    main()
