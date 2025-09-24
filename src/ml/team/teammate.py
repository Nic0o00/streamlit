import os
import re
import spacy
import pandas as pd

# Charger spaCy (anglais)
nlp = spacy.load("en_core_web_sm")

# Chemins
input_dir = 'data/processed/translated'
OUT_FILE = 'output/predictions/team.csv'
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# Mots pour filtrer les advisors
ADVISOR_TERMS = {"advisor","mentor","board member","investor"}

# Rôles connus (C-level limité à 1, autres multiples)
ROLE_PATTERNS = {
    r'\bCEO\b': "CEO",
    r'\bCFO\b': "CFO",
    r'\bCOO\b': "COO",
    r'\bCTO\b': "CTO",
    r'\bCSO\b': "CSO",
    r'\bCMO\b': "CMO",
    r'\b[\w\s-]*engineer(s)?\b': "Engineer",
    r'\b[\w\s-]*developer(s)?\b': "Developer",
    r'\b[\w\s-]*scientist(s)?\b': "Scientist",
    r'\b[\w\s-]*designer(s)?\b': "Designer",
    r'\b[\w\s-]*analyst(s)?\b': "Analyst",
    r'\b[\w\s-]*doctor(s)?\b': "Doctor",
    r'\b[\w\s-]*researcher(s)?\b': "Researcher",
    r'\bproduct manager(s)?\b': "Product Manager",
    r'\bmarketing\b': "Marketing",
    r'\bsales\b': "Sales",
    r'\bHR\b': "HR",
}

C_LEVEL = {"CEO","CFO","COO","CTO","CSO","CMO"}

# Compter les founders dans le texte
def count_founders_and_roles(text):
    doc = nlp(text)
    founders = set()
    roles_found = []

    # Découper en phrases pour éviter doublons
    sentences = list(doc.sents)
    
    for sent in sentences:
        sent_text = sent.text.lower()

        # 1) détecter PERSON et check founder
        for ent in sent.ents:
            if ent.label_ == "PERSON" and not any(term in sent_text for term in ADVISOR_TERMS):
                # si phrase contient founder/co-founder → marquer comme founder
                if re.search(r'\b(co-?founder|founder)\b', sent_text):
                    founders.add(ent.text.strip())
        
        # 2) détecter rôles
        sent_roles = set()
        for pattern, role in ROLE_PATTERNS.items():
            if re.search(pattern, sent.text, re.IGNORECASE):
                sent_roles.add(role)

        for role in sent_roles:
            if role in C_LEVEL:
                if role not in roles_found:  # max 1 occurrence global
                    roles_found.append(role)
            else:
                roles_found.append(role)  # multiples ok

    # fallback: si aucun founder détecté via NER, compter occurrences founder
    if not founders:
        count = len(re.findall(r'\b(co-?founder|founder)\b', text, re.IGNORECASE))
        return max(1,count), roles_found

    return len(founders), roles_found

# Parcourir tous les fichiers
docs, founders_count_list, postes_list = [], [], []

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        founders_count, roles_found = count_founders_and_roles(content)

        docs.append(filename)
        founders_count_list.append(founders_count)
        postes_list.append(", ".join(roles_found) if roles_found else "")

# Sauvegarde CSV
out = pd.DataFrame({
    "deck": docs,
    "founders_count": founders_count_list,
    "postes": postes_list
})

out.to_csv(OUT_FILE, sep=";", index=False)
print("Predictions saved to", OUT_FILE)
