"""
Module de traitement de texte PDF.

Ce module permet d'extraire le texte brut d'un PDF et de le normaliser.
La normalisation inclut la mise en minuscules et la suppression
des caractères spéciaux, ne conservant que les lettres, chiffres,
accents français et tirets.
"""

import fitz
import re

def normalize_text(text):
    """
    Normalise un texte donné.

    Parameters
    ----------
    text : str
        Le texte à normaliser.

    Returns
    -------
    str
        Le texte normalisé, en minuscules et sans caractères spéciaux.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâéèêôùç\- ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF et le normalise.

    Parameters
    ----------
    pdf_path : str
        Chemin vers le fichier PDF.

    Returns
    -------
    str
        Texte extrait et normalisé.
    
    Notes
    -----
    - Chaque page du PDF est parcourue et son texte ajouté.
    - Le texte est ensuite passé à `normalize_text` pour nettoyage.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text_clean = normalize_text(text)
    return text_clean
