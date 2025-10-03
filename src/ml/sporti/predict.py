import os
import joblib
import pandas as pd

# chemins
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "..", "..", "models", "deck_classifier.joblib")
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LABELED_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
OUTPUT_CSV = os.path.join(BASE, "..", "..", "..", "output", "predictions", "tfidf_vectors_with_resultat_predictions.csv")

def predict_all_decks():
    # charger modèle
    clf = joblib.load(MODEL_PATH)
    
    # charger vecteurs TF-IDF avec colonne 'doc'
    X = pd.read_csv(VECT_CSV, sep=";")

    # supprimer les vecteurs sans colonne doc (si jamais)
    X = X.dropna(subset=["doc"]).reset_index(drop=True)

    # garder la colonne 'doc' pour aligner les résultats
    doc_names = X["doc"]

    # vecteurs pour prédiction
    X_vectors = X.drop(columns=["doc"])

    # prédiction
    preds = clf.predict(X_vectors)
    probs = clf.predict_proba(X_vectors).max(axis=1)

    # dataframe résultat avec seulement les colonnes demandées
    results = pd.DataFrame({
        "doc": doc_names,
        "predicted_resultat": preds,
        "confidence_score": probs
    })

    # sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="ISO-8859-1")
    print(f"Résultats sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    predict_all_decks()
