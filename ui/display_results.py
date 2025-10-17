import streamlit as st
import pandas as pd
import os

PREDICTIONS_DIR = os.path.join("output", "predictions")  # ou passer en param

def afficher_resultat_deck(deck_name):
    try:
        files = ["tfidf_vectors_with_tech_predictions.csv",
                 "tfidf_vectors_with_domain_predictions.csv",
                 "tfidf_vectors_with_country_predictions.csv",
                 "tfidf_vectors_with_resultat_predictions.csv"]
        file_paths = [os.path.join(PREDICTIONS_DIR, f) for f in files]
        if not all(os.path.exists(f) for f in file_paths):
            st.warning("⚠️ Certains fichiers de prédictions sont manquants.")
            return

        tech_df = pd.read_csv(file_paths[0], sep=";")
        domain_df = pd.read_csv(file_paths[1], sep=";")
        country_df = pd.read_csv(file_paths[2], sep=";")
        resultat_df = pd.read_csv(file_paths[3], sep=";")

        merged_df = tech_df.merge(domain_df, on="doc") \
                           .merge(country_df, on="doc") \
                           .merge(resultat_df, on="doc")

        row = merged_df[merged_df["doc"] == deck_name]
        if row.empty:
            st.warning(f"Aucun résultat trouvé pour **{deck_name}**.")
            return
        row = row.iloc[0]

        st.subheader(f"Résultats pour `{deck_name}`")
        pays_ok = row["predicted_country"].lower() in ["benelux", "france", "germany"]
        domaine_ok = row["predicted_domain"].lower() in [
            "energy transition", "others", "industry 4.0", "new materials"
        ]
        statut = "✅ IN" if (pays_ok and domaine_ok) else "❌ OUT"

        st.markdown(f"### 🏁 Statut scope : **{statut}**")
        st.write(f"**Technologie :** {row['predicted_tech']}")
        st.write(f"**Domaine :** {row['predicted_domain']}")
        st.write(f"**Pays :** {row['predicted_country']}")
        st.write(f"**Résultat :** {row['predicted_resultat']}")

    except Exception as e:
        st.error(f"Erreur lors du chargement du résultat : {e}")
