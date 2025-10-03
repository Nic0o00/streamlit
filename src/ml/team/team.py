import os
import re
import json
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TXT_DIR = os.path.join(BASE_DIR, "data", "processed", "translated")
OUTPUT = os.path.join(BASE_DIR, "output", "predictions")
os.makedirs(OUTPUT, exist_ok=True)

# Regex clés
FOUNDER_ROLES = re.compile(r"\b(founder|co[- ]?founder|founding team|founder/ceo)\b", re.I)
FOUNDERS_LIST = re.compile(r"\bFounders?\s*[:\-]\s*(.+)", re.I)  # ex: "Founders: John, Jane"
NAME_ROLE = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*[-–,:]\s*(.+)", re.I)  # ex: "John Doe - CEO"

def normalize_name(name: str) -> str:
    """Nettoie et met en forme les noms (title case)."""
    return " ".join([w.capitalize() for w in name.strip().split()])

def extract_founders(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    slides = [s.strip() for s in content.split("---slide---") if s.strip()]
    founders = []

    for idx, slide in enumerate(slides, start=1):
        # 1. Cas "Founders: John Doe, Jane Smith"
        m = FOUNDERS_LIST.search(slide)
        if m:
            names = re.split(r"[;,/&]|\band\b", m.group(1))
            for n in names:
                n = normalize_name(n)
                if len(n.split()) >= 2:  # garder seulement "Prénom Nom"
                    founders.append({"name": n, "role": "Founder", "slide": idx})

        # 2. Cas "Nom – Founder"
        for m in NAME_ROLE.finditer(slide):
            name, role = m.groups()
            if FOUNDER_ROLES.search(role):
                founders.append({"name": normalize_name(name), "role": "Founder", "slide": idx})

        # 3. spaCy fallback (si mot-clé founder présent dans le slide)
        if FOUNDER_ROLES.search(slide):
            doc = nlp(slide)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    founders.append({"name": normalize_name(ent.text), "role": "Founder", "slide": idx})

    # Déduplication
    seen = set()
    unique = []
    for f in founders:
        if f["name"].lower() not in seen:
            seen.add(f["name"].lower())
            unique.append(f)

    return unique

def process_all_decks(txt_dir):
    rows = []
    for filename in sorted(os.listdir(txt_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(txt_dir, filename)
        founders = extract_founders(path)
        rows.append({
            "deck_name": filename,
            "num_founders": len(founders),
            "founders": json.dumps(founders, ensure_ascii=False)
        })
        print(f"[{filename}] Founders: {len(founders)}")

    out_csv = os.path.join(OUTPUT, "founders_composition.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print("[OK] Sauvegardé :", out_csv)

if __name__ == "__main__":
    process_all_decks(TXT_DIR)
