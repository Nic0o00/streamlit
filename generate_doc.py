"""
Script pour générer la documentation HTML de tous les modules du projet
et la placer dans le dossier `doc/`.
"""

import os
import subprocess


# Répertoire racine du projet (dossier où se trouve ce script)
root_dir = os.path.dirname(os.path.abspath(__file__))

# Répertoire où stocker la doc
docs_dir = os.path.join(root_dir, "doc", "html")
os.makedirs(docs_dir, exist_ok=True)

# Liste des modules à documenter (notation Python)
modules = [
    "src.ml.country.train_country",
    "src.ml.country.predict_country",
    "src.ml.domain.train_domain",
    "src.ml.domain.predict_domain",
    "src.ml.tech.train_tech",
    "src.ml.tech.predict_tech",
    "src.treatment.detect_lang",
    "src.treatment.extract_text",
    "src.treatment.translate",
    "src.vectorisation.vectorize_text"
]


for mod in modules:
    #print(f"Génération de la doc pour {mod} ")
    # Exécution de pydoc pour générer le HTML directement dans doc/
    subprocess.run(
        ["python", "-m", "pydoc", "-w", mod],
          # le HTML sera créé dans ce dossier
    )
    html_file = f"{mod}.html"

for file in os.listdir(root_dir):
    if file.endswith(".html"):
        src_path = os.path.join(root_dir, file)
        dest_path = os.path.join(docs_dir, file)
        if os.path.exists(dest_path):
            os.remove(dest_path)  # Supprime l'ancien fichier
        os.rename(src_path, dest_path)
        print(f"{file} déplacé vers {docs_dir}")


index_file = os.path.join(root_dir, "doc", "index.html")
with open(index_file, "w", encoding="utf-8") as f:
    f.write("<!DOCTYPE html>\n<html lang='fr'>\n<head>\n")
    f.write("<meta charset='UTF-8'>\n<title>Documentation Projet</title>\n</head>\n<body>\n")
    f.write("<h1>Documentation du projet</h1>\n<ul>\n")
    for file in sorted(os.listdir(docs_dir)):
        if file.endswith(".html"):
            # lien relatif vers le sous-dossier html/
            f.write(f"<li><a href='html/{file}'>{file}</a></li>\n")
    f.write("</ul>\n</body>\n</html>\n")
  
print("Toutes les documentations ont été générées dans le dossier doc/")
