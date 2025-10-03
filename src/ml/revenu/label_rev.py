import os
import pandas as pd
import requests
import time

# === Config API ===
HF_API_TOKEN = "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"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
API_URL = f"https://router.huggingface.co/v1/chat/completions/{MODEL_ID}"
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# === Fichiers et dossiers ===
input_dir = "./data/processed/translated/"
tech_file = "./output/predictions/tfidf_vectors_with_tech_predictions.csv"
output_csv = "./output/predictions/deck_revenue_types_api.csv"
allowed_types = ["licensing", "recurring", "one time"]

# === Charger le mapping hard/soft/both ===
tech_df = pd.read_csv(tech_file, sep=";")
tech_df['doc_name'] = tech_df['doc'].apply(lambda x: os.path.splitext(x)[0])
tech_map = dict(zip(tech_df['doc_name'], tech_df['predicted_tech']))

def query_hf(prompt):
    payload = {
        "inputs": prompt,
        "options": {
            "use_cache": False,
            "wait_for_model": True    # attend que le modèle soit prêt si nécessaire
        },
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.0
        }
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    if resp.status_code == 200:
        j = resp.json()
        # dépend de la structure de retour 
        # on suppose que c'est une liste avec generated_text, ou équivalent
        if isinstance(j, list) and "generated_text" in j[0]:
            return j[0]["generated_text"].strip().lower()
        # ou autre champ selon l’API
        elif "generated_text" in j:
            return j["generated_text"].strip().lower()
        else:
            # si pas le champ, peut être un champ "text" ou "output"
            # imprime j pour debug
            print("⚠ réponse inattendue:", j)
            return ""
    else:
        print(f"⚠ Erreur API {resp.status_code}: {resp.text}")
        return ""

results = []

for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(input_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print(f"Fichier vide : {filename}")
        continue

    deck_name = os.path.splitext(filename)[0]
    prod_type = tech_map.get(deck_name, "both")

    prompt = (
        "You are an expert analyzing startup pitch decks. "
        "Classify the revenue model of the following startup as one of these categories ONLY:\n"
        "- recurring: subscription-based, SaaS, memberships, or other recurring payments.\n"
        "- licensing: revenue from software licenses, patents, or intellectual property.\n"
        "- one time: single purchase, upfront payment, or one-off fees.\n\n"
        f"The product type is '{prod_type}'. Only select allowed categories based on this type:\n"
        "  * soft → recurring or licensing\n"
        "  * hard → recurring or one time\n"
        "  * both → any of the three\n\n"
        f"Startup description:\n{text}\n\n"
        "Answer ONLY with one word: licensing, recurring, or one time."
    )

    prediction = query_hf(prompt)
    if not prediction or prediction not in allowed_types:
        # fallback logique selon prod_type
        if prod_type == "soft":
            prediction = "recurring"
        elif prod_type == "hard":
            prediction = "one time"
        else:
            prediction = "recurring"

    results.append({
        "deck": filename,
        "revenue_type": prediction,
        "product_type": prod_type
    })

    # Pour ne pas surcharger l’API / respecter les quotas
    time.sleep(1)  

# Export CSV
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ CSV généré : {output_csv}")
