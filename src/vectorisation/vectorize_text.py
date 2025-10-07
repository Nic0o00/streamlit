"""
Module : vectorize_text
=======================

Ce module permet de générer des vecteurs TF-IDF enrichis pour les documents
texte présents dans le dossier `data/processed/translated/`.

Fonctions principales :
-----------------------
- `vectorize_text()` : lit tous les fichiers texte, construit les vecteurs TF-IDF
  (1-grammes, 2-grammes, 3-grammes) et sauvegarde le DataFrame résultant dans
  `data/processed/tfidf_vectors.csv`.

Paramètres TF-IDF utilisés :
----------------------------
- ngram_range=(1,3) : prend en compte unigrams, bigrams et trigrams
- max_features=7000  : limite le nombre de termes pour éviter l’explosion mémoire
- min_df=2           : ignore les termes trop rares
"""

import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text():
    """
    Lit tous les fichiers texte du dossier `translated`, calcule les vecteurs
    TF-IDF enrichis et sauvegarde le résultat dans un CSV.

    Étapes :
    --------
    1. Lecture de tous les fichiers .txt dans `data/processed/translated/`
    2. Stockage dans un DataFrame avec colonnes : "doc" et "text"
    3. Construction d'un vecteur TF-IDF avec ngram_range=(1,3)
    4. Conversion en DataFrame avec les termes comme colonnes
    5. Ajout de la colonne "doc" en première position
    6. Sauvegarde du DataFrame TF-IDF dans `data/processed/tfidf_vectors.csv`

    Raises
    ------
    ValueError
        Si aucun fichier .txt n'est trouvé dans le dossier source.
    """

    # --- Définition des chemins ---
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "..", "..", "data", "processed", "translated")
    output_file = os.path.join(base_dir, "..", "..", "data", "processed", "tfidf_vectors.csv")

    # --- Lecture des fichiers texte ---
    docs = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            path = os.path.join(input_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # On conserve le nom du doc en ajoutant .pdf pour cohérence
            docs.append({"doc": os.path.splitext(fname)[0] + ".pdf", "text": text})

    if not docs:
        raise ValueError("Aucun fichier .txt trouvé dans le dossier translated.")

    df = pd.DataFrame(docs)
    df["doc"] = df["doc"].astype(str)
    print(f"{len(df)} documents chargés depuis {input_dir}")

    # --- Création du vecteur TF-IDF enrichi ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1,3),   # unigrammes, bigrammes et trigrammes
        max_features=7000,   # limite pour éviter l'explosion mémoire
        min_df=2             # ignorer termes trop rares
    )
    X = vectorizer.fit_transform(df["text"].astype(str))

    # --- Conversion en DataFrame ---
    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # --- Ajouter la colonne "doc" en première position ---
    tfidf_df.insert(0, "doc", df["doc"])

    # --- Sauvegarde du CSV TF-IDF ---
    tfidf_df.to_csv(output_file, sep=";", index=False)
    st.success(f"Vecteurs TF-IDF enrichis sauvegardés")


# --- Point d'entrée principal ---
if __name__ == "__main__":
    vectorize_text()
