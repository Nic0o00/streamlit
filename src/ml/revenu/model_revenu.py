import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# === Fichiers ===
features_file = "./data/processed/tfidf_vectors.csv"  # ton fichier TF-IDF
labels_file = "./output/predictions/deck_revenue_types.csv"  # ton fichier labels générés
model_file = "./models/revenue_model.pkl"

# === Charger les features ===
X = pd.read_csv(features_file)
docs = X["doc"]
X = X.drop(columns=["doc"])

# === Charger les labels ===
labels_df = pd.read_csv(labels_file)

# Harmoniser noms (deck vs doc → sans extension)
labels_df["deck_name"] = labels_df["deck"].apply(lambda x: os.path.splitext(x)[0])
docs_clean = docs.apply(lambda x: os.path.splitext(x)[0])

# Créer mapping doc -> label
labels_map = dict(zip(labels_df["deck_name"], labels_df["revenue_type"]))

# Aligner y avec X
y = docs_clean.map(labels_map)

# Supprimer les lignes sans label ou "unknown"
mask = y.notna() & (y != "unknown")
X = X[mask]
y = y[mask]
docs = docs[mask]

print(f"Données alignées : {X.shape[0]} exemples, {X.shape[1]} features")

# === Entraînement Logistic Regression ===
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X, y)

# === Sauvegarde modèle ===
os.makedirs(os.path.dirname(model_file), exist_ok=True)
joblib.dump(clf, model_file)
print(f"✅ Modèle sauvegardé : {model_file}")
