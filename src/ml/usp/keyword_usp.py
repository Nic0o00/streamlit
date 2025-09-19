"""
Module pour extraire automatiquement les mots-clés USP
à partir des fichiers texte bruts des decks.
"""

import os
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

# Télécharger les stopwords anglais si nécessaire
nltk.download('stopwords')

def extract_usp_keywords_from_texts(data_dir, top_n):
    """
    Parcourt tous les fichiers .txt dans data_dir
    et retourne un DataFrame avec colonnes :
    'doc' et 'usp_keywords' (liste des top-N mots les plus fréquents)
    """
    english_stopwords = set(stopwords.words('english'))
    texts = []
    doc_names = []

    # Parcours des fichiers .txt
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                if len(text) < 10:
                    print(f"⚠️ Ignored {filename}: texte trop court")
                    continue
                texts.append(text)
                doc_names.append(filename)

    # Extraction des mots-clés
    usp_keywords_list = []
    for text in texts:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        tokens = [t for t in tokens if t not in english_stopwords]
        counts = Counter(tokens)
        top_keywords = [word for word, _ in counts.most_common(top_n)]
        top_keywords += [""] * (top_n - len(top_keywords))  # compléter si moins de mots
        usp_keywords_list.append(top_keywords)

    # DataFrame final
    df_keywords = pd.DataFrame({
        "doc": doc_names,
        "usp_keywords": usp_keywords_list
    })
    return df_keywords

# ---------------------------
# Exemple d'exécution
# ---------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "..", "data", "processed","translated")
    df_keywords = extract_usp_keywords_from_texts(data_dir, top_n=10)
    print(df_keywords.head())
