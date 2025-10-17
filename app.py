import streamlit as st
import os
from ui.display_results import afficher_resultat_deck
from ui.upload import upload_and_process_files
from ui.predictions import lancer_vectorisation_et_predictions

BASE_DIR = os.path.dirname(__file__)
DECKS_DIR = os.path.join(BASE_DIR, "data", "decks")
TRANSLATED_DIR = os.path.join(BASE_DIR, "data", "processed", "translated")

uploaded_file_names = upload_and_process_files(DECKS_DIR, TRANSLATED_DIR)
# --- Bouton vectorisation et prédictions (seulement si des fichiers uploadés) ---
if uploaded_file_names:  # seulement s'il y a des fichiers uploadés
    if st.button("🚀 Lancer vectorisation et prédictions"):
        lancer_vectorisation_et_predictions(uploaded_file_names)
        st.session_state["vectorisation_done"] = True
else:
    st.info("📂 Téléversez au moins un fichier PDF pour lancer la vectorisation et les prédictions.")

# Barre latérale pour les anciens decks
st.sidebar.header("📂 Decks disponibles")
all_decks = [f for f in os.listdir(DECKS_DIR) if f.endswith(".pdf")]
if all_decks:
    selected_deck = st.sidebar.selectbox("Choisir un deck :", ["—"] + all_decks)
    if selected_deck != "—":
        if st.sidebar.button("👁️ Voir le résultat"):
            st.session_state["deck_a_afficher"] = selected_deck

# Affichage simulé pop-up
if "deck_a_afficher" in st.session_state:
    st.markdown("---")
    with st.expander(f"📊 Résultats pour {st.session_state['deck_a_afficher']}", expanded=True):
        afficher_resultat_deck(st.session_state["deck_a_afficher"])
        if st.button("Fermer la fenêtre"):
            del st.session_state["deck_a_afficher"]

# Bouton pour voir les résultats des fichiers uploadés
if st.session_state.get("vectorisation_done", False):
    if st.button("📊 Voir les résultats des fichiers uploadés"):
        for name in uploaded_file_names:
            with st.expander(f"📄 Résultat : {name}", expanded=True):
                afficher_resultat_deck(name)
