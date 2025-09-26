import pandas as pd
import joblib
import os

# === Fichiers ===
features_file = "./data/processed/tfidf_vectors.csv" 
model_file = "./models/revenue_model.pkl"
output_csv = "./output/predictions/deck_revenue_types_from_logreg.csv"

# === Charger features ===
X = pd.read_csv(features_file)
docs = X["doc"]
X = X.drop(columns=["doc"])

# === Charger modèle ===
clf = joblib.load(model_file)

# === Prédictions ===
preds = clf.predict(X)

# === Export résultat ===
df = pd.DataFrame({
    "deck": docs,
    "revenue_type_pred": preds
})
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"Résultats exportés dans {output_csv}")
