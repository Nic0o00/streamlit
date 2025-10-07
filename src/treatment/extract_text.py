import fitz
import re
import io

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâéèêôùç\- ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_pdf(pdf_source):
    """
    Extrait le texte d'un PDF (depuis un chemin ou un fichier uploadé) 
    et ajoute un séparateur de slide.
    
    Parameters
    ----------
    pdf_source : str | BytesIO
        - Chemin du fichier PDF (ex: "docs/rapport.pdf")
        - ou fichier uploadé Streamlit (objet UploadedFile)
    
    Returns
    -------
    str
        Texte complet du PDF normalisé, avec "---slide---" comme séparateur entre les pages.
    """
    # --- Gestion des deux types d'entrée ---
    if isinstance(pdf_source, str):
        # Cas 1 : chemin classique
        doc = fitz.open(pdf_source)
    else:
        # Cas 2 : fichier en mémoire (upload Streamlit)
        pdf_source.seek(0)
        doc = fitz.open(stream=pdf_source.read(), filetype="pdf")

    # --- Extraction du texte ---
    text = ""
    for page in doc:
        page_text = page.get_text()
        text += page_text + "\n---slide---\n"

    # --- Normalisation ---
    text_clean = normalize_text(text)
    return text_clean
