import os
import pandas as pd
from src.ml.domain.model_domain import train_domain
from src.ml.tech.model_tech import train_tech
from src.ml.country.model_country import train_country
from src.ml.resultat.model_resultat import train_resultat


BASE_DIR = os.path.dirname(__file__)
LABELED_CSV = os.path.join(BASE_DIR, "data", "labeled.csv")

def get_new_decks(decks_dir, labeled_csv=LABELED_CSV):
    """
    Retourne la liste des fichiers PDF dans decks_dir qui ne sont pas encore labellisés.
    """
    # Récupérer les fichiers déjà labellisés
    if os.path.exists(labeled_csv):
        labeled_df = pd.read_csv(labeled_csv, sep=";")
        # on ne garde que le nom de fichier sans chemin
        labeled_files = labeled_df["doc"].apply(lambda x: os.path.basename(x)).tolist()
    else:
        labeled_df = pd.DataFrame(columns=["doc", "tech", "domain", "country", "resultat"])
        labeled_files = []

    # Récupérer tous les fichiers PDF dans le dossier decks_dir
    all_decks = [f for f in os.listdir(decks_dir) if f.endswith(".pdf")]

    # Ne garder que ceux qui ne sont pas déjà labellisés
    new_decks = [f for f in all_decks if f not in labeled_files]

    return new_decks, labeled_df


def add_and_train(new_rows, labeled_csv=LABELED_CSV):
    """
    Ajoute les nouvelles lignes validées dans labeled.csv et entraîne les modèles.
    new_rows : liste de dictionnaires avec les colonnes : doc, tech, domain, country, resultat
    """
    # Charger CSV existant ou créer
    if os.path.exists(labeled_csv):
        df = pd.read_csv(labeled_csv, sep=";")
    else:
        df = pd.DataFrame(columns=["doc", "tech", "domain", "country", "resultat"])

    # Ajouter les nouvelles lignes
    new_df = pd.DataFrame(new_rows, columns=["doc", "tech", "domain", "country", "resultat"])
    df = pd.concat([df, new_df], ignore_index=True)

    # Sauvegarder
    df.to_csv(labeled_csv, sep=";", index=False)

    # Entraîner les modèles
    train_domain(df)
    train_tech(df)
    train_country(df)
    train_resultat(df)

    return len(new_rows)
