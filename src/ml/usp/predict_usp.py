"""
Script pour générer automatiquement des USP (Unique Selling Propositions)
à partir de fichiers texte de pitch decks.
"""

import os
import pandas as pd
from transformers import pipeline

# ---------------------------
# Charger le modèle Hugging Face
# ---------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # tu peux mettre flan-t5-base si ça rame trop
    device=-1  # -1 = CPU, change en 0 si tu as un GPU
)

def generate_usp(text: str) -> str:
    """
    Génère une USP concise à partir d'un texte brut de pitch deck.
    """
    prompt = (
        "You are a marketing expert. A Unique Selling Proposition (USP) "
        "is a short, clear statement (one sentence) that explains what makes a company "
        "or product different and why customers should choose it. "
        "From the following startup pitch, extract the USP, "
        "focusing on the unique value and customer benefit:\n\n"
        f"{text}\n\nUSP:"
    )
    try:
        result = generator(prompt, max_length=60, do_sample=False)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

def process_directory(data_dir: str, output_csv: str):
    """
    Parcourt tous les .txt d'un dossier, génère une USP pour chacun,
    et sauvegarde dans un CSV.
    """
    docs, usps = [], []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if len(text) < 20:
                print(f"Ignored {filename}: texte trop court")
                continue

            usp = generate_usp(text)
            docs.append(filename)
            usps.append(usp)
            print(f"{filename} → {usp}")

    df = pd.DataFrame({"doc": docs, "USP": usps})
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n🎯 Résultats sauvegardés dans {output_csv}")

# ---------------------------
# Exemple d'exécution
# ---------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "..", "data", "processed", "translated")
    output_csv = os.path.join(base_dir, "..", "..", "..", "output", "usp_predictions.csv")

    process_directory(data_dir, output_csv)
