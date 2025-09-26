import os
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# === Modèle gratuit Hugging Face ===
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# === Dossiers et fichiers ===
input_dir = "./data/processed/translated/"
output_csv = "./output/predictions/deck_revenue_types.csv"
allowed_types = ["licensing", "recurring", "one time"]

# === Lire le fichier indiquant hard/soft/both ===
tech_file = "./output/predictions/tfidf_vectors_with_tech_predictions.csv"
tech_df = pd.read_csv(tech_file, sep=";")

# Créer un mapping basé sur le nom du deck sans extension
tech_df['doc_name'] = tech_df['doc'].apply(lambda x: os.path.splitext(x)[0])
tech_map = dict(zip(tech_df['doc_name'], tech_df['predicted_tech']))

results = []

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Fichier vide : {filename}")
            continue

        # Récupérer type produit (hard/soft/both)
        deck_name = os.path.splitext(filename)[0]
        prod_type = tech_map.get(deck_name, "both")

        # Prompt enrichi
        prompt = (
                    "You are an expert analyzing startup pitch decks. "
                    "Classify the revenue model of the following startup as one of these categories ONLY:\n"
                    "- recurring: subscription-based, SaaS, memberships, or other recurring payments.\n"
                    "- licensing: revenue from software licenses, patents, or intellectual property.\n"
                    "- one time: single purchase, upfront payment, or one-off fees.\n\n"
                    f"The product type is '{prod_type}'. Only select allowed categories based on this type.\n\n"
                    f"Startup description:\n{text}\n\n"
                    "Answer ONLY with one word: licensing, recurring, or one time."
                )



        # Tokenize et générer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=10)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

        

    
        results.append({
            "deck": filename,
            "revenue_type": prediction,
            "product_type": prod_type
        })

# Export CSV
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"CSV generated: {output_csv} \n\n Fin de la prédictions des revenus")
