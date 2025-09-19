"""
Point d'entrée pour générer les USP pour tous les decks.
Utilise les mots-clés extraits et FLAN-T5-large pour générer des phrases USP.
"""

import os
import pandas as pd
from keyword_usp import extract_usp_keywords_from_texts
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ---------------------------
# Charger le modèle FLAN-T5-large
# ---------------------------
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ---------------------------
# Fonction de génération USP à partir des mots-clés
# ---------------------------
def generate_usp_from_keywords(keywords):
    keywords = [k for k in keywords if k]
    if not keywords:
        return "USP could not be generated"

    # Prompt explicite pour guider la génération
    prompt = (
        f"You are an expert startup analyst. Using the following keywords, "
        f"write a single, concise, clear Unique selling proposition (USP) sentence of the startup's main value proposition to present to an investor. "
        f"Focus only on products, services, or innovations. "
        f"I want a short sentence between 10 and 20 words presenting the USP. "
        f"Do NOT include names, numbers, emails, or legal/confidential info. "
        f"Keywords: {', '.join(keywords)}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    usp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return usp

# ---------------------------
# Configuration dossiers
# ---------------------------
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "..", "..", "data", "processed","translated")
output_dir = os.path.join(base_dir, "..", "..", "..","output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "usp_predictions.csv")

# ---------------------------
# Étape 1 : extraire les mots-clés
# ---------------------------
top_n_keywords = 5
df_keywords = extract_usp_keywords_from_texts(data_dir, top_n=top_n_keywords)

# ---------------------------
# Étape 2 : générer la phrase USP
# ---------------------------
df_keywords["usp_sentence"] = df_keywords["usp_keywords"].apply(generate_usp_from_keywords)

# ---------------------------
# Étape 3 : sauvegarder les résultats
# ---------------------------
df_keywords.to_csv(output_file, index=False)
print("✅ USP generation completed. Preview:")
print(df_keywords.head())
