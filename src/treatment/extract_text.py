"""
Module : extract_text_from_pdf
==============================

Ce module permet d'extraire et de normaliser le texte de fichiers PDF.
Il utilise la librairie `PyMuPDF` (fitz) pour lire les PDF et ajoute des
séparateurs de slides.

Fonctions principales :
-----------------------
- `normalize_text(text)` : normalise le texte en minuscules, supprime les
  caractères spéciaux et réduit les espaces multiples.
- `extract_text_from_pdf(pdf_path)` : extrait le texte d'un PDF et applique
  la normalisation, en ajoutant un séparateur "---slide---" entre les pages.
"""

import fitz
import re

def normalize_text(text):
    """
    Normalise le texte.

    Étapes de normalisation :
    -------------------------
    1. Passage en minuscules
    2. Suppression des caractères spéciaux (ne garde que lettres, chiffres, accents et tirets)
    3. Remplacement des multiples espaces par un seul espace
    4. Suppression des espaces en début et fin de texte

    Parameters
    ----------
    text : str
        Texte à normaliser.

    Returns
    -------
    str
        Texte normalisé.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9àâéèêôùç\- ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un PDF et ajoute un séparateur de slide.

    Parameters
    ----------
    pdf_path : str
        Chemin vers le fichier PDF à traiter.

    Returns
    -------
    str
        Texte complet du PDF normalisé, avec "---slide---" comme séparateur
        entre les pages.
    
    Notes
    -----
    - Utilise `fitz` pour lire les pages du PDF.
    - Chaque page est concaténée avec un retour à la ligne et le marqueur de slide.
    - Le texte final est ensuite normalisé via `normalize_text`.
    """
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        page_text = page.get_text()
        text += page_text + "\n---slide---\n"  # marqueur de slide

    text_clean = normalize_text(text)
    return text_clean
