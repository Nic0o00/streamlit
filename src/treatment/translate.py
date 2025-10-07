"""
Module : translate_text
=======================

Ce module fournit une fonction pour traduire automatiquement un texte
vers une langue cible à l'aide de l'API Google Translator, avec gestion
des textes longs et des erreurs.

Fonctions principales :
-----------------------
- `translate_text(text, target='en', max_chunk_size=4500)` :
  Traduit un texte, découpe les segments trop longs, et retourne le texte
  original si une erreur survient.
"""

from deep_translator import GoogleTranslator

def translate_text(text, target='en', max_chunk_size=4500):
    """
    Traduit un texte vers une langue cible, en découpant si nécessaire.

    Parameters
    ----------
    text : str
        Le texte source à traduire.
    target : str, optional
        Code de la langue cible (par défaut 'en' pour anglais).
    max_chunk_size : int, optional
        Taille maximale d'un segment à traduire d'un coup (par défaut 4500 caractères).

    Returns
    -------
    str
        Texte traduit. Si la traduction échoue, retourne le texte original.

    Notes
    -----
    - Les textes sont découpés en segments pour gérer les limitations de l'API.
    - Utilise `deep_translator.GoogleTranslator` avec détection automatique de la langue source.
    - Les erreurs sont capturées : le texte original est retourné en cas d'échec.
    """
    try:
        translator = GoogleTranslator(source='auto', target=target)

        # --- Traduction directe si texte court ---
        if len(text) <= max_chunk_size:
            return translator.translate(text)

        # --- Découpage du texte en segments pour éviter les limites de l'API ---
        segments = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_segments = [translator.translate(segment) for segment in segments]

        # --- Assemblage des segments traduits ---
        translated_text = ' '.join(translated_segments)
        return translated_text

    except Exception as e:
        # --- Gestion des erreurs : afficher message et retourner texte original ---
        print(f"[ERREUR TRADUCTION] {e}")
        return text
