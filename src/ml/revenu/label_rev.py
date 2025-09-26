import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Modèle Mistral-7B-Instruct ===
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # modèle Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # utilise GPU si disponible
    torch_dtype=torch.float16 # pour réduire la VRAM utilisée
)

# === Dossiers et fichiers ===
input_dir = "./data/processed/translated/"
output_csv = "./output/predictions/deck_revenue_types.csv"
allowed_types = ["licensing", "recurring", "one time"]

# === Lire le fichier indiquant hard/soft/both ===
tech_file = "./output/predictions/tfidf_vectors_with_tech_predictions.csv"
tech_df = pd.read_csv(tech_file, sep=";")
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

        deck_name = os.path.splitext(filename)[0]
        prod_type = tech_map.get(deck_name, "both")

        # === Prompt enrichi ===
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

        # === Tokenize et génération ===
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

        # === Nettoyage du résultat ===
        if prediction not in allowed_types:
            prediction = "unknown"

        results.append({
            "deck": filename,
            "revenue_type": prediction,
            "product_type": prod_type
        })

# === Export CSV ===
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"CSV generated: {output_csv}")
