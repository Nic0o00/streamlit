"""
Module de génération TF-IDF enrichi pour la technologie des documents.

Lit tous les fichiers texte de `data/translated/`, construit des vecteurs
TF-IDF (1-grammes, 2-grammes, 3-grammes) et sauvegarde dans
`data/processed/tfidfvectorizer_tech.csv`.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text():
    # --- Chemins ---
    base_dir = os.path.dirname(__file__)
    input_dir  = os.path.join(base_dir, "..", "..", "data", "processed", "translated")
    output_file = os.path.join(base_dir, "..", "..", "data", "processed", "tfidf_vectors.csv")

    # --- Lire tous les fichiers texte ---
    docs = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            path = os.path.join(input_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append({"doc": os.path.splitext(fname)[0] + ".pdf", "text": text})


    if not docs:
        raise ValueError("Aucun fichier .txt trouvé dans le dossier translated.")

    df = pd.DataFrame(docs)
    df["doc"] = df["doc"].astype(str)  # assure que doc est bien str
    print(f"{len(df)} documents chargés depuis {input_dir}")

    # --- TF-IDF enrichi ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1,3),     # unigrammes + bigrammes + trigrammes
        max_features=7000,    # limite pour éviter explosion mémoire
        min_df=2,              # ignorer termes trop rares
        #max_df=0.8             # ignorer termes trop fréquents
    )
    X = vectorizer.fit_transform(df["text"].astype(str))

    # --- Conversion DataFrame ---
    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

   # Assure que 'doc' est bien str
    df["doc"] = df["doc"].astype(str)

    # Crée le TF-IDF
    X = vectorizer.fit_transform(df["text"].astype(str))
    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Si doc existe déjà, on le supprime pour éviter doublon
    if "doc" in tfidf_df.columns:
        tfidf_df = tfidf_df.drop(columns=["doc"])

    # Ajouter doc en première colonne
    tfidf_df.insert(0, "doc", df["doc"])


    # --- Sauvegarde ---
    tfidf_df.to_csv(output_file, sep=";", index=False)
    print(f"Vecteurs TF-IDF enrichis sauvegardés dans : {output_file}")


if __name__ == "__main__":
    vectorize_text()
