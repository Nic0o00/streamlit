"""
Module de traduction de texte avec gestion de grands textes.

Ce module utilise l'API Google Translator pour traduire automatiquement
un texte vers une langue cible, en découpant les textes trop longs
pour éviter les limites de taille de l'API. Le module est robuste et
retourne le texte original en cas d'erreur.
"""

from deep_translator import GoogleTranslator

def translate_text(text, target='en', max_chunk_size=4500):
    """
    Traduit un texte vers la langue cible, avec découpage si nécessaire.

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
        Le texte traduit. Si la traduction échoue, retourne le texte original.

    Notes
    -----
    - Le texte est découpé en segments pour gérer les textes trop longs.
    - Le module utilise `deep_translator.GoogleTranslator` avec détection automatique de la langue source.
    - En cas d'erreur lors de la traduction, un message est affiché et le texte original est retourné.
    """

    try:
        translator = GoogleTranslator(source='auto', target=target)

        # --- Traduction directe si texte court ---
        if len(text) <= max_chunk_size:
            return translator.translate(text)

        # --- Découpage du texte en segments pour éviter les limites ---
        segments = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_segments = [translator.translate(segment) for segment in segments]

        # --- Assemblage des segments traduits ---
        translated_text = ' '.join(translated_segments)
        return translated_text

    except Exception as e:
        # --- Gestion des erreurs : retourne le texte original ---
        print(f"[ERREUR TRADUCTION] {e}")
        return text
