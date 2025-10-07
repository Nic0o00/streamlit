"""
Module : detect_language
=======================

Ce module fournit une fonction pour détecter automatiquement la langue
d'un texte à l'aide de la librairie `langdetect`.

Fonctions principales :
-----------------------
- `detect_language(text)` : détecte le code langue (ex: 'en', 'fr', 'de')
  d'un texte donné, ou retourne `None` si la langue ne peut pas être détectée.
"""

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Variable globale optionnelle pour stocker le dernier résultat
result = None

def detect_language(text):
    """
    Détecte la langue d'un texte donné.

    Paramètres
    ----------
    text : str
        Le texte dont on souhaite détecter la langue.

    Retour
    ------
    str or None
        - Code de la langue détectée (ex: 'en', 'fr', 'de')
        - `None` si la langue ne peut pas être détectée ou si le texte est vide.

    Notes
    -----
    - Les textes vides ou contenant uniquement des espaces retournent `None`.
    - Les exceptions levées par la librairie `langdetect` sont interceptées et
      retournent également `None`.
    """
    global result

    # --- Vérification des textes vides ---
    if not text or len(text.strip()) == 0:
        result = None  # Texte vide ou uniquement espaces

    try:
        # --- Détection de la langue ---
        lang = detect(text)
        result = lang
    except LangDetectException:
        # --- Gestion des erreurs de détection ---
        result = None

    return result
