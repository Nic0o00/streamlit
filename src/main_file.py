import os
from treatment.extract_text import extract_text_from_pdf
from treatment.detect_lang import detect_language
from treatment.translate import translate_text

from vectorisation.vectorize_text import vectorize_text

#from ml.model_domain import predict_domain
from ml.techno.predict_techno import predict_tech
from ml.domain.predict_domain import predict_domain
from ml.country.predict_country import predict_country

from ml.evaluate import evaluate




DECK_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "decks")
TRANSLATED_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "translated")

os.makedirs(TRANSLATED_DIRECTORY, exist_ok=True)

docs = []
texts = []

for each_file in os.listdir(DECK_DIRECTORY):
    deck_path = os.path.join(DECK_DIRECTORY, each_file)
    translated_path = os.path.join(TRANSLATED_DIRECTORY, each_file.replace(".pdf", ".txt"))

    if os.path.exists(translated_path):
        with open(translated_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = extract_text_from_pdf(deck_path)         # Extraction texte brut
        lang = detect_language(text)                     # Détection langue
        if lang != "en":
            text = translate_text(text)                  # Traduction automatique
        if text is None:
            text = "none"
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(text)

    docs.append(each_file)      # Nom du fichier
    texts.append(text)          # Texte (traduit si nécessaire)


# --- Vectorisation TF-IDF globale ---
vectorize_text()

# --- Predictions ---
predict_tech()

predict_domain()

predict_country()

evaluate("tech")
evaluate("domain")
evaluate("country")

