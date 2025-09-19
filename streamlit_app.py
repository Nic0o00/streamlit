import streamlit as st
import os
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.detect_lang import detect_language
from src.treatment.translate import translate_text
from src.vectorisation.vectorize_text import vectorize_text
from src.ml.techno.predict_techno import predict_tech
from src.ml.domain.predict_domain import predict_domain
from src.ml.country.predict_country import predict_country
from src.ml.evaluate import evaluate

# --- Config directories ---
BASE_DIR = os.path.dirname(__file__)
DECK_DIRECTORY = os.path.join(BASE_DIR, "data", "decks")
TRANSLATED_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", "translated")
os.makedirs(TRANSLATED_DIRECTORY, exist_ok=True)

st.title("Analyse de Pitch Decks")

# --- Sidebar options ---
st.sidebar.header("Options")
uploaded_files = st.sidebar.file_uploader("Uploader des fichiers PDF", type="pdf", accept_multiple_files=True)
run_all = st.sidebar.button("Lancer l'analyse globale")

# --- Fonction pour traiter un fichier PDF ---
def process_pdf(file, save_translated=True):
    text = extract_text_from_pdf(file)
    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text)
    if text is None:
        text = "none"
    if save_translated:
        translated_path = os.path.join(TRANSLATED_DIRECTORY, os.path.basename(file.name))
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text

# --- Traitement uploadé ---
if uploaded_files:
    st.subheader("Résultats par fichier")
    for uploaded_file in uploaded_files:
        st.markdown(f"### {uploaded_file.name}")
        text = process_pdf(uploaded_file)
        st.text_area("Texte traduit", text, height=300)

        # Predictions
        st.markdown("**Prédictions :**")
        tech_pred = predict_tech([text])[0]
        domain_pred = predict_domain([text])[0]
        country_pred = predict_country([text])[0]

        st.write(f"- Technologie : {tech_pred}")
        st.write(f"- Domaine : {domain_pred}")
        st.write(f"- Pays : {country_pred}")

# --- Analyse globale à partir du dossier DECK_DIRECTORY ---
if run_all:
    st.subheader("Analyse globale du dossier")
    docs = []
    texts = []

    for each_file in os.listdir(DECK_DIRECTORY):
        deck_path = os.path.join(DECK_DIRECTORY, each_file)
        translated_path = os.path.join(TRANSLATED_DIRECTORY, each_file.replace(".pdf", ".txt"))

        if os.path.exists(translated_path):
            with open(translated_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = extract_text_from_pdf(deck_path)
            lang = detect_language(text)
            if lang != "en":
                text = translate_text(text)
            if text is None:
                text = "none"
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(text)

        docs.append(each_file)
        texts.append(text)

    st.write(f"{len(docs)} fichiers traités")

    # Vectorisation globale
    vectorize_text()

    # Predictions
    predict_tech()
    predict_domain()
    predict_country()

    # Évaluations
    st.write("### Évaluations")
    st.write("Tech : ", evaluate("tech"))
    st.write("Domain : ", evaluate("domain"))
    st.write("Country : ", evaluate("country"))

    st.success("Analyse globale terminée !")

