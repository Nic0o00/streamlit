import os
import re
import pandas as pd

# Chemins
input_dir = 'data/processed/translated'
OUT_FILE = 'output/predictions/team.csv'

# Créer le dossier de sortie si nécessaire
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# Fonction pour compter les occurrences de "founder" / "co-founder"
def count_founders(text):
    matches = re.findall(r'\b(co-?founder|founder)\b', text, re.IGNORECASE)
    return len(matches) if matches else 1  # par défaut 1 si rien trouvé

# Parcourir tous les fichiers txt
docs = []
founders_count_list = []

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        count = count_founders(content)
        docs.append(filename)
        founders_count_list.append(count)

# Sauvegarder avec pandas
out = pd.DataFrame({
    "deck": docs,
    "founders_count": founders_count_list
})

out.to_csv(OUT_FILE, sep=";", index=False)
print("Predictions saved to", OUT_FILE)
