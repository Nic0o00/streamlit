import streamlit as st
import os
import fitz  # PyMuPDF
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.detect_lang import detect_language
from src.treatment.translate import translate_text
from src.vectorisation.vectorize_text import vectorize_text
from src.ml.tech.predict_tech import predict_tech
from src.ml.domain.predict_domain import predict_domain
from src.ml.country.predict_country import predict_country
from src.ml.resultat.predict_resultat import predict_resultat

# --- Configuration de la page ---
st.set_page_config(page_title="ScoringApp - PDF Reader", layout="wide")
st.title("📊 ScoringApp - Lecture et analyse de documents")
st.markdown("Téléversez vos fichiers PDF : ils seront enregistrés, traduits et vectorisés de façon persistante.")

# --- Dossiers ---
BASE_DIR = os.path.dirname(__file__)
DECKS_DIRECTORY = os.path.join(BASE_DIR,"data", "decks")
TRANSLATED_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", "translated")

os.makedirs(DECKS_DIRECTORY, exist_ok=True)
os.makedirs(TRANSLATED_DIRECTORY, exist_ok=True)

# --- Upload multiple PDF ---
uploaded_files = st.file_uploader(
    "📂 Choisissez un ou plusieurs fichiers PDF à traiter :",
    type=["pdf"],
    accept_multiple_files=True
)

# --- Traitement ---
if uploaded_files:
    st.info(f"{len(uploaded_files)} fichier(s) uploadé(s).")

    with st.expander("📘 Voir les fichiers uploadés et leur contenu", expanded=True):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DECKS_DIRECTORY, uploaded_file.name)
            txt_path = os.path.join(
                TRANSLATED_DIRECTORY, uploaded_file.name.replace(".pdf", ".txt")
            )

            st.subheader(f"📄 {uploaded_file.name}")

            # --- Étape 1 : Sauvegarde PDF s’il n’existe pas déjà ---
            if os.path.exists(file_path):
                st.warning(f"⚠️ Le fichier `{uploaded_file.name}` existe déjà dans `data/decks`. Il ne sera pas remplacé.")
            else:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"💾 Fichier enregistré dans `{file_path}`")

            # --- Étape 2 : Vérifier si le texte existe déjà ---
            if os.path.exists(txt_path):
                st.info(f"📝 Le texte traduit existe déjà : `{txt_path}`")
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                try:
                    # Extraction du texte
                    uploaded_file.seek(0)
                    text = extract_text_from_pdf(uploaded_file)

                    # Détection de la langue
                    lang = detect_language(text)
                    st.write(f"🌍 Langue détectée : **{lang}**")

                    # Traduction automatique
                    if lang != "en":
                        st.info("🔁 Traduction du texte en anglais...")
                        text = translate_text(text)

                    if not text:
                        text = "Aucun texte détecté."

                    # Sauvegarde du texte
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    st.success(f"✅ Texte sauvegardé dans `{txt_path}`")

                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement de {uploaded_file.name} : {e}")
                    continue  # passe au suivant

            # --- Étape 3 : Affichage du texte ---
            with st.expander(f"🔍 Aperçu du contenu : {uploaded_file.name}"):
                st.text_area("", text, height=250)

    # --- Étape 4 : Vectorisation globale ---
    # --- Étape 4 : Vectorisation globale ---
    if st.button("🚀 Lancer la vectorisation de tous les fichiers traduits"):
        st.info("⚙️ Vectorisation en cours...")
        vectorize_text()
        st.success("✅ Vectorisation terminée avec succès !")

        # --- Étape 5 : Prédictions ---
        st.info("🔮 Prédiction de la technologie...")
        predict_tech()
        st.success("✅ Prédiction de la technologie terminée !")

        st.info("🔮 Prédiction du domaine...")
        predict_domain()
        st.success("✅ Prédiction du domaine terminée !")

        st.info("🔮 Prédiction du pays...")
        predict_country()
        st.success("✅ Prédiction du pays terminée !")

        st.info("🔮 Prédiction du résultat...")
        predict_resultat()
        st.success("✅ Prédiction du résultat terminée !")


else:
    st.warning("Veuillez téléverser au moins un fichier PDF pour commencer.")
