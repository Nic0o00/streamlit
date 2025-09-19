"""
USP Generation with Hugging Face Instruct Models (Mistral/LLaMA)
"""

import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Nettoyage du texte
# ---------------------------
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)  # emails
    text = re.sub(r'http\S+|www\.\S+', '', text)  # urls
    text = re.sub(r'\b\d+\b', '', text)  # chiffres isolés
    text = re.sub(r'COPYRIGHT.*|Confidential.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)  # espaces multiples
    return text.strip()

# ---------------------------
# Charger modèle Hugging Face
# ---------------------------
# ⚠️ Mets ici le modèle que tu veux tester : 
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "meta-llama/Llama-2-7b-chat-hf"
# (les poids se téléchargent automatiquement)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"🔄 Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU si dispo, CPU sinon
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.7,
    top_p=0.9
)

# ---------------------------
# Génération USP
# ---------------------------
def generate_usp_from_text(text):
    text = clean_text(text)
    if not text or len(text) < 20:
        return "USP could not be generated"

    prompt = (
        "A Unique Selling Proposition (USP) is a single, short, persuasive sentence "
        "that explains why a customer should choose this startup over competitors.\n"
        "It must highlight the UNIQUE benefit for the customer, not just describe the product.\n\n"
        f"Startup deck text:\n{text}\n\n"
        "Write ONE clear USP sentence (under 25 words):"
    )

    outputs = generator(prompt)
    usp = outputs[0]["generated_text"].split("USP sentence:")[-1].strip()
    return usp

# ---------------------------
# Pipeline fichiers
# ---------------------------
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "..", "..", "data", "processed", "translated")
output_dir = os.path.join(base_dir, "..", "..", "..", "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "usp_predictions.csv")

deck_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
results = []

for filename in deck_files:
    path = os.path.join(data_dir, filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    usp_sentence = generate_usp_from_text(text)
    results.append({"doc": filename, "usp_sentence": usp_sentence})
    print(f"[OK] Generated USP for {filename}: {usp_sentence}")

df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)
print(f"✅ USP generation completed. Results saved in {output_file}")
print(df_results.head())
