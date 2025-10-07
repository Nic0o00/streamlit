"""
Script de fusion de fichiers CSV contenant des informations sur des deals et des documents.

Ce script a pour objectif de :
1. Charger deux fichiers CSV (un contenant des métadonnées de documents, l’autre des informations de deals).
2. Nettoyer et harmoniser les colonnes nécessaires à la fusion.
3. Réaliser une jointure entre les deux jeux de données sur des noms de documents/deals nettoyés.
4. Sélectionner les colonnes finales pertinentes.
5. Sauvegarder le résultat fusionné dans un nouveau CSV.

Entrées :
    - doc1.csv : Fichier contenant des informations sur les deals.
    - labeled.csv : Fichier contenant la liste des documents et leurs métadonnées.
Sortie :
    - merged.csv : Fichier fusionné contenant le nom du document et le type de deal associé.
"""

import os
import pandas as pd

# --- Définition des chemins vers les fichiers sources et de sortie ---
BASE = os.path.dirname(__file__)
DOC1_CSV = os.path.join(BASE, "..", "..", "..", "doc1.csv")  # Fichier des deals
DOC2_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")  # Fichier des documents labellisés
OUTPUT_CSV = os.path.join(BASE, "..", "..", "..", "merged.csv")  # Fichier de sortie

# --- Chargement des fichiers CSV ---
# Lecture des deux fichiers avec le séparateur ";" et encodage ISO-8859-1
df1 = pd.read_csv(DOC1_CSV, sep=";", encoding="ISO-8859-1")
df2 = pd.read_csv(DOC2_CSV, sep=";", encoding="ISO-8859-1")

# --- Nettoyage des noms de colonnes ---
# Suppression des espaces superflus autour des noms de colonnes
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# --- Harmonisation des identifiants pour la fusion ---
# Nettoyage et uniformisation des identifiants (suppression des espaces et mise en minuscule)
# Objectif : pouvoir comparer les noms de deals et les noms de documents sans erreur de casse ou d'espaces
df1["Deal_name_clean"] = (
    df1["Deal name"]
    .str.replace(" ", "", regex=False)
    .str.lower()
)
df2["doc_clean"] = (
    df2["doc"]
    .str.replace(".pdf", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.lower()
)

# --- Fusion des deux DataFrames ---
# On conserve toutes les lignes du fichier 'labeled' (left join)
merged = pd.merge(
    df2, df1,
    left_on="doc_clean",
    right_on="Deal_name_clean",
    how="left"
)

# --- Sélection des colonnes finales ---
# Seules les colonnes utiles à l’analyse sont conservées
final_df = merged[["doc", "Deal type"]]

# --- Sauvegarde du CSV fusionné ---
# Le fichier final est enregistré avec le même encodage et séparateur
final_df.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="ISO-8859-1")

# --- Message de confirmation ---
print(f"✅ CSV fusionné sauvegardé dans {OUTPUT_CSV} ({len(final_df)} lignes)")
