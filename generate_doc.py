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

# Liste complète des modules à documenter
modules = [
    #ML
    "src.ml.country.model_country",
    "src.ml.country.predict_country",

    "src.ml.domain.model_domain",
    "src.ml.domain.predict_domain",

    "src.ml.resultat.model_resultat",
    "src.ml.resultat.predict_resultat",

    "src.ml.tech.model_tech",
    "src.ml.tech.predict_tech",

    # Evaluation
    "src.ml.evaluate",

    # Vectorisation
    "src.vectorisation.vectorize_text",

    # Text treatment
    "src.treatment.detect_lang",
    "src.treatment.extract_text",
    "src.treatment.translate",
]

# Génération des docs HTML
for mod in modules:
    print(f"📄 Génération doc pour {mod} …")
    result = subprocess.run(["python", "-m", "pydoc", "-w", mod], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Échec génération pour {mod} : {result.stderr}")
        continue

    # Le fichier généré est mod_name.html dans le cwd
    html_file = f"{mod}.html"
    src_path = os.path.join(root_dir, html_file)

    if not os.path.exists(src_path):
        print(f"Fichier HTML non trouvé pour {mod}, peut-être module non importable.")
        continue

    dest_path = os.path.join(docs_dir, html_file)
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(src_path, dest_path)
    print(f"{html_file} déplacé vers {docs_dir}")

# Création d'un index HTML
index_file = os.path.join(root_dir, "doc", "index.html")
with open(index_file, "w", encoding="utf-8") as f:
    f.write("<!DOCTYPE html>\n<html lang='fr'>\n<head>\n")
    f.write("<meta charset='UTF-8'>\n<title>Documentation Projet</title>\n</head>\n<body>\n")
    f.write("<h1>Documentation du projet</h1>\n<ul>\n")
    for file in sorted(os.listdir(docs_dir)):
        if file.endswith(".html"):
            f.write(f"<li><a href='html/{file}'>{file}</a></li>\n")
    f.write("</ul>\n</body>\n</html>\n")

print("Toutes les documentations ont été générées dans le dossier doc/")
