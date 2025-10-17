import streamlit as st
from difflib import SequenceMatcher

def compare_texts(existing_text, new_text, filename):
    """
    Affiche les deux textes côte à côte, calcule un pourcentage de similarité
    et demande à l'utilisateur de choisir :
    - Écraser l'ancien
    - Renommer le nouveau
    - Annuler
    Retourne le choix sous forme de chaîne après validation.
    """
    st.markdown(f"### Comparaison pour `{filename}`")

    # Calcul du pourcentage de similarité
    similarity_ratio = SequenceMatcher(None, existing_text, new_text).ratio()
    similarity_percent = round(similarity_ratio * 100, 2)
    st.info(f"📊 Similarité entre l'ancien et le nouveau texte : **{similarity_percent}%**")

    # Affichage côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Texte existant**")
        st.text_area("Ancien texte", existing_text, height=300)
    with col2:
        st.markdown("**Texte du nouveau PDF**")
        st.text_area("Nouveau texte", new_text, height=300)

    # Choix utilisateur
    choice = st.radio(
        "Que voulez-vous faire ?",
        ("Écraser", "Renommer", "Annuler"),
        key=f"compare_{filename}"
    )

    # Bouton de validation
    validated_choice = None
    if st.button("✅ Valider le choix", key=f"validate_{filename}"):
        validated_choice = choice
        st.success(f"Vous avez choisi : **{validated_choice}**")

    return validated_choice
