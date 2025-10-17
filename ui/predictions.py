import streamlit as st
from src.vectorisation.vectorize_text import vectorize_text
from src.ml.tech.predict_tech import predict_tech
from src.ml.domain.predict_domain import predict_domain
from src.ml.country.predict_country import predict_country
from src.ml.resultat.predict_resultat import predict_resultat

def lancer_vectorisation_et_predictions(uploaded_file_names):
    if st.button("🚀 Lancer vectorisation et prédictions"):
        if uploaded_file_names:
            with st.spinner("⏳ Traitement des fichiers..."):
                vectorize_text()
                predict_tech()
                predict_domain()
                predict_country()
                predict_resultat()
            st.success(f"✅ Vectorisation et prédictions terminées pour {len(uploaded_file_names)} fichier(s)")
            st.session_state["vectorisation_done"] = True
        else:
            st.warning("Aucun fichier à traiter.")
