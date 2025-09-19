"""
Module de détection automatique de la langue d'un texte.

Ce module utilise la librairie `langdetect` pour identifier la langue d'un texte donné.
Il gère les textes vides ou non détectables et capture les exceptions liées à
la détection de langue.

Fonctions
---------
detect_language(text)
    Détecte la langue d'un texte.
"""
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

result = None

def detect_language(text):
    """
    Détecte la langue d'un texte donné.

    Parameters
    ----------
    text : str
        Le texte dont on souhaite détecter la langue.

    Returns
    -------
    str or None
        Code de la langue détectée (ex: 'en', 'fr', 'de'), ou `None` si la langue
        ne peut pas être détectée ou si le texte est vide.

    Notes
    -----
    - Les textes vides ou contenant uniquement des espaces retournent `None`.
    - Les exceptions levées par la librairie `langdetect` sont interceptées et
      retournent également `None`.
    """
    if not text or len(text.strip()) == 0:
        result = None  # Ou "unknown", ou une autre valeur signifiant absence de langue détectable
    try:
        lang = detect(text)
        result = lang
    except LangDetectException:
        result = None

    return result
