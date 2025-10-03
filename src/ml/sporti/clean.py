import os
import pandas as pd

# chemins vers tes fichiers
BASE = os.path.dirname(__file__)
DOC1_CSV = os.path.join(BASE, "..", "..", "..", "doc1.csv")
DOC2_CSV = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
OUTPUT_CSV = os.path.join(BASE, "..", "..", "..", "merged.csv")

# charger CSV
df1 = pd.read_csv(DOC1_CSV, sep=";", encoding="ISO-8859-1")
df2 = pd.read_csv(DOC2_CSV, sep=";", encoding="ISO-8859-1")

# nettoyer les colonnes
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# harmoniser pour merge
df1["Deal_name_clean"] = df1["Deal name"].str.replace(" ", "", regex=False).str.lower()
df2["doc_clean"] = df2["doc"].str.replace(".pdf", "", regex=False).str.replace(" ", "", regex=False).str.lower()

# merge : conserver toutes les lignes de labeled
merged = pd.merge(df2, df1, left_on="doc_clean", right_on="Deal_name_clean", how="left")

# sélectionner uniquement les colonnes souhaitées
final_df = merged[["doc", "Deal type"]]

# sauvegarder CSV final
final_df.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="ISO-8859-1")
print(f"✅ CSV fusionné sauvegardé dans {OUTPUT_CSV} ({len(final_df)} lignes)")

