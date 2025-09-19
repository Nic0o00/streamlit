import os
import streamlit as st
import pandas as pd

from treatment.extract_text import extract_text_from_pdf
from treatment.detect_lang import detect_language
from treatment.translate import translate_text
from vectorisation.vectorize_text import vectorize_text
from ml.techno.predict_techno import predict_tech
from ml.domain.predict_domain import predict_domain
from ml.country.predict_country import predict_country

# --- Dossiers ---
TRANSLATED_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "processed", "translated")
os.makedirs(TRANSLATED_DIRECTORY, exist_ok=True)

st.title("Analyse automatique de pitch decks")

# --- Upload PDF ---
uploaded_files = st.file_uploader("Choisir un ou plusieurs PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        # --- Extraction texte ---
        translated_path = os.path.join(TRANSLATED_DIRECTORY, uploaded_file.name.replace(".pdf", ".txt"))

        if os.path.exists(translated_path):
            with open(translated_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = extract_text_from_pdf(uploaded_file)
            lang = detect_language(text)
            if lang != "en":
                text = translate_text(text)
            if text is None:
                text = "none"
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(text)

        # --- Vectorisation ---
        # Pour l'instant on suppose que vectorize_text() vectorise tout automatiquement
        vectorize_text()  # Si nécessaire, adapter pour accepter du texte individuel

        # --- Prédictions ---
        tech_pred = predict_tech([text])[0]
        domain_pred = predict_domain([text])[0]
        country_pred = predict_country([text])[0]

        # --- Stocker résultats ---
        results.append({
            "document": uploaded_file.name,
            "tech": tech_pred,
            "domain": domain_pred,
            "country": country_pred,
        })

    # --- Affichage ---
    df_results = pd.DataFrame(results)
    st.subheader("Résultats des prédictions")
    st.dataframe(df_results)
