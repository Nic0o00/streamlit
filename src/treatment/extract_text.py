import fitz
import re
from io import BytesIO

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâéèêôùç\- ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    """
    Extrait le texte d'un PDF uploadé via Streamlit et le normalise.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Fichier PDF uploadé via st.file_uploader()

    Returns
    -------
    str
        Texte extrait et normalisé.
    """
    # Convertir le fichier uploadé en flux mémoire
    pdf_bytes = uploaded_file.read()
    pdf_stream = BytesIO(pdf_bytes)

    # Ouvrir le PDF depuis le flux mémoire
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    text_clean = normalize_text(text)
    return text_clean
