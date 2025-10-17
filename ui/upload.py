import os
import streamlit as st
from src.treatment.extract_text import extract_text_from_pdf
from src.treatment.detect_lang import detect_language
from src.treatment.translate import translate_text
from ui.compare import compare_texts

def upload_and_process_files(decks_dir, translated_dir):
    uploaded_files = st.file_uploader(
        "📂 Choisissez un ou plusieurs fichiers PDF à traiter :",
        type=["pdf"],
        accept_multiple_files=True
    )

    uploaded_file_names = []

    if uploaded_files:
        st.info(f"{len(uploaded_files)} fichier(s) sélectionné(s).")

        for uploaded_file in uploaded_files:
            file_path = os.path.join(decks_dir, uploaded_file.name)
            txt_path = os.path.join(translated_dir, uploaded_file.name.replace(".pdf", ".txt"))

            # Extraire le texte du PDF
            try:
                uploaded_file.seek(0)
                new_text = extract_text_from_pdf(uploaded_file)
                lang = detect_language(new_text)
                if lang != "en":
                    new_text = translate_text(new_text)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'extraction de {uploaded_file.name} : {e}")
                continue

            key_prefix = uploaded_file.name.replace(".", "_")

            # Cas fichier déjà existant
            if os.path.exists(file_path):
                st.warning(f"⚠️ Le fichier `{uploaded_file.name}` existe déjà.")

                if f"{key_prefix}_option" not in st.session_state:
                    st.session_state[f"{key_prefix}_option"] = None

                # Radio pour choix utilisateur
                option = st.radio(
                    f"Que faire pour `{uploaded_file.name}` ?",
                    ("Écraser", "Renommer", "Comparer avant décision"),
                    key=f"{key_prefix}_radio"
                )
                st.session_state[f"{key_prefix}_option"] = option

                # Si comparaison
                similarity = None
                if option == "Comparer avant décision":
                    existing_text = ""
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            existing_text = f.read()
                    similarity = compare_texts(existing_text, new_text, uploaded_file.name)
                    st.info(f"💡 Similarité : {similarity:.2f}%" if similarity is not None else "")

                # Bouton de validation
                if st.button(f"Valider le choix pour {uploaded_file.name}", key=f"btn_{key_prefix}"):
                    choice = st.session_state[f"{key_prefix}_option"]

                    if choice == "Écraser":
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(new_text)
                        st.success(f"💾 `{uploaded_file.name}` écrasé.")
                        uploaded_file_names.append(uploaded_file.name)

                    elif choice == "Renommer" or (choice == "Comparer avant décision" and st.session_state.get(f"{key_prefix}_decision") == "Renommer"):
                        i = 1
                        new_name = uploaded_file.name.replace(".pdf", f"_{i}.pdf")
                        while os.path.exists(os.path.join(decks_dir, new_name)):
                            i += 1
                            new_name = uploaded_file.name.replace(".pdf", f"_{i}.pdf")
                        new_file_path = os.path.join(decks_dir, new_name)
                        new_txt_path = os.path.join(translated_dir, new_name.replace(".pdf", ".txt"))
                        with open(new_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with open(new_txt_path, "w", encoding="utf-8") as f:
                            f.write(new_text)
                        st.success(f"💾 Nouveau fichier enregistré sous `{new_name}`")
                        uploaded_file_names.append(new_name)

                    elif choice == "Comparer avant décision":
                        st.session_state[f"{key_prefix}_decision"] = "Écraser"  # ou "Renommer", l'utilisateur peut re-sélectionner
                        st.info("💡 Choisissez Écraser ou Renommer puis validez à nouveau.")

            else:
                # Fichier n'existe pas, sauvegarde directe
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                st.success(f"💾 `{uploaded_file.name}` enregistré.")
                uploaded_file_names.append(uploaded_file.name)

            # Aperçu du texte
            with st.expander(f"🔍 Aperçu : {uploaded_file.name}", expanded=False):
                st.text_area("", new_text, height=250)

    return uploaded_file_names
