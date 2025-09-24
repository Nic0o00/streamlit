import os
import re
import spacy
import pandas as pd

# Charger le modèle spaCy anglais médium
nlp = spacy.load("en_core_web_md")

input_folder = "data/processed/translated"
output_file = "output/predictions/team.csv"

# Définition des rôles
TEAM_ROLES = [
    "ceo", "cto", "cfo", "coo",
    "engineer", "developer", "advisor", "chairman", "board", 
    "executive", "head of product", "product manager", 
    "sales manager", "marketing manager", "lead", "director",
    "dr"
]

FOUNDER_KEYWORDS = ["founder", "co-founder", "founding team"]

# Rôles pouvant apparaître plusieurs fois dans la composition
MULTIPLE_ROLES = ["engineer", "developer", "advisor", "dr"]

# Rôles uniques légaux pour le compte
UNIQUE_ROLES = ["ceo", "cto", "cfo", "coo"]

def find_team_count(text, person_roles, cofounder_names):
    """
    Calcul du nombre total de membres de l'équipe avec règles :
    - Rôles uniques légaux comptent 1 max
    - Rôles multiples comptent toutes occurrences
    - Cofounders comptent pour 1 chacun
    - Ignorer les nombres >60
    """
    text_lower = text.lower()
    numbers = []

    patterns = [
        r"(\d+)\s*team members",
        r"team members[:\s]*(\d+)",
        r"(\d+)\s*employees",
        r"(\d+)\s*more employees"
    ]

    for pat in patterns:
        matches = re.findall(pat, text_lower)
        for m in matches:
            n = int(m)
            if n <= 60:
                numbers.append(n)

    # Compter les personnes détectées
    total_people = 0
    counted_roles = set()

    # Rôles uniques légaux
    for role in UNIQUE_ROLES:
        for r in person_roles.values():
            if r.lower() == role and role not in counted_roles:
                total_people += 1
                counted_roles.add(role)
                break

    # Rôles multiples
    for r in person_roles.values():
        if r.lower() in MULTIPLE_ROLES:
            total_people += 1

    # Cofounders
    total_people += len(cofounder_names)

    # Comparer avec les nombres détectés dans le texte
    if numbers:
        total_people = max(total_people, max(numbers))

    return total_people

def extract_team_info(text):
    text_lower = text.lower()
    doc = nlp(text)

    person_roles = {}        # nom -> rôle principal
    cofounder_names = set()  # noms des fondateurs uniques

    # 1️⃣ Repérer les contextes "founder"
    founder_contexts = []
    for match in re.finditer(r"(founder|co-founder|founding team)", text_lower):
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        founder_contexts.append(text_lower[start:end])

    # 2️⃣ Parcours des entités PERSON
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            name_lower = name.lower()

            # Détection cofounder
            for context in founder_contexts:
                if name_lower in context:
                    cofounder_names.add(name)
                    break

            # Détection des rôles autour du nom
            window = text_lower.split()
            name_tokens = name_lower.split()
            idx = [i for i, w in enumerate(window) if w in name_tokens]
            for i in idx:
                local_window = window[max(0, i-5): i+6]
                local_text = " ".join(local_window)

                for role in TEAM_ROLES:
                    if role.lower() in local_text and name not in person_roles:
                        person_roles[name] = role
                        break

    # 3️⃣ Nombre total de membres
    team_count = find_team_count(text, person_roles, cofounder_names)
    cofounder_count = len(cofounder_names)

    # 4️⃣ Composition
    composition_roles = []
    seen_unique_roles = set()
    for role in person_roles.values():
        if role.lower() in MULTIPLE_ROLES:
            composition_roles.append(role)
        else:
            if role.lower() not in seen_unique_roles:
                composition_roles.append(role)
                seen_unique_roles.add(role.lower())

    return cofounder_count, team_count, composition_roles

# Lecture des fichiers et création du CSV
data = []
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            cofounder_count, team_count, composition = extract_team_info(text)
            data.append({
                "deck": filename,
                "cofounder": 1 if cofounder_count == 0 and team_count!=0 else cofounder_count,
                "team": team_count,
                "composition": ", ".join(composition)
            })

df = pd.DataFrame(data)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)

print(f"Team info saved to {output_file}")
