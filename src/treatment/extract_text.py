import fitz
import re

def normalize_text(text):
    """
    Normalise le texte : minuscules, retire les caractères spéciaux,
    remplace les espaces multiples par un seul espace.
    """
    text = re.sub(r'[^a-z0-9àâéèêôùç\- ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un PDF depuis son chemin et ajoute des séparateurs de slide.
    
    Parameters
    ----------
    pdf_path : str
        Chemin vers le fichier PDF.
    
    Returns
    -------
    str
        Texte normalisé avec un séparateur "---slide---" entre les pages.
    """
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        page_text = page.get_text()
        text += page_text + "\n---slide---\n"  # marqueur de slide

    text_clean = normalize_text(text)
    return text_clean
