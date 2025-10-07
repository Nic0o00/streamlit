"""
Module : evaluate_predictions
=============================

Ce module permet d’évaluer les prédictions d’un label donné par rapport aux labels
réels dans le fichier `labeled_total.csv`.

Fonction principale :
--------------------
- `evaluate(label_col)` : compare les prédictions et les labels, calcule l’accuracy,
  affiche la répartition des classes, les documents mal prédits et la matrice de confusion.

Exemple d’utilisation :
-----------------------
>>> evaluate("tech")
>>> evaluate("domain")
>>> evaluate("country")

Sorties :
---------
- Affichage console des métriques et des erreurs
- Retourne un tuple (df_eval, df_wrong, cm_df) :
    - `df_eval` : DataFrame fusionné avec colonne "correct"
    - `df_wrong` : DataFrame des documents mal prédits
    - `cm_df` : matrice de confusion au format DataFrame
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate(label_col):
    """
    Évalue les prédictions pour un label donné en comparant avec les labels réels.

    Paramètres
    ----------
    label_col : str
        Nom du label à évaluer (ex: "tech", "domain", "country", "resultat").

    Étapes
    -------
    1. Chargement du fichier de prédictions et du fichier `labeled_total.csv`
    2. Harmonisation des colonnes "doc" pour jointure
    3. Fusion des prédictions et des labels réels sur "doc"
    4. Exclusion des valeurs vides ou NaN
    5. Calcul de l’accuracy et du nombre de prédictions correctes/incorrectes
    6. Affichage de la répartition des classes réelles et prédites
    7. Affichage des documents mal prédits
    8. Calcul et affichage de la matrice de confusion

    Retour
    ------
    tuple
        - df_eval : DataFrame fusionné avec colonne "correct"
        - df_wrong : DataFrame des documents mal prédits
        - cm_df : matrice de confusion (DataFrame)
    """

    # --- Définition des chemins de fichiers ---
    pred_file = f"tfidf_vectors_with_{label_col}_predictions.csv"
    predict_col = f"predicted_{label_col}"
    base_dir = os.path.dirname(__file__)
    path_pred  = os.path.join(base_dir, "..", "..", "output", "predictions", pred_file)
    path_label = os.path.join(base_dir, "..", "..", "labeled_total.csv")

    print(f"\n=== Évaluation pour {label_col} ===")
    print(f"Fichier prédictions : {path_pred}")
    print(f"Fichier labels      : {path_label}")

    # --- Lecture des fichiers CSV ---
    df_pred = pd.read_csv(path_pred, sep=";")
    df_label = pd.read_csv(path_label, sep=";")

    # --- Harmonisation des colonnes "doc" ---
    df_pred["doc"] = df_pred["doc"].astype(str)
    df_label["doc"] = df_label["doc"].astype(str)

    # --- Jointure prédictions / labels réels ---
    df_eval = pd.merge(
        df_pred[["doc", predict_col]],
        df_label[["doc", label_col]],
        on="doc", how="inner"
    )

    if len(df_eval) == 0:
        raise ValueError(f"Aucune correspondance entre prédictions et labels pour {label_col}.")

    # --- Suppression des valeurs vides ---
    df_eval = df_eval.dropna(subset=[predict_col, label_col])
    df_eval = df_eval[(df_eval[predict_col] != "") & (df_eval[label_col] != "")]

    # --- Calcul colonne "correct" ---
    df_eval["correct"] = df_eval[predict_col] == df_eval[label_col]

    total = len(df_eval)
    correct = df_eval["correct"].sum()
    wrong = total - correct
    accuracy = correct / total * 100

    # --- Résultats globaux ---
    print(f"Total docs évalués : {total}")
    print(f"Bons prédits       : {correct}")
    print(f"Faux prédits       : {wrong}")
    print(f"Accuracy (%)       : {accuracy:.2f}%")

    # --- Répartition des classes ---
    print("\n--- Répartition des classes réelles ---")
    print(df_eval[label_col].value_counts(normalize=True))

    print("\n--- Répartition des classes prédites ---")
    print(df_eval[predict_col].value_counts(normalize=True))

    # --- Documents mal prédits ---
    df_wrong = df_eval[~df_eval["correct"]][["doc", predict_col, label_col]]
    if len(df_wrong) > 0:
        print("\n--- Documents mal prédits ---")
        print(df_wrong.to_string(index=False))
    else:
        print("\nAucun document mal prédit.")

    # --- Matrice de confusion ---
    df_eval[predict_col] = df_eval[predict_col].astype(str)
    df_eval[label_col] = df_eval[label_col].astype(str)
    labels_sorted = sorted(df_eval[label_col].unique())
    cm = confusion_matrix(df_eval[label_col], df_eval[predict_col], labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print("\n--- Matrice de confusion ---")
    print(cm_df)

    # --- Retourne les résultats pour analyse externe ---
    return df_eval, df_wrong, cm_df


# --- Point d’entrée principal ---
if __name__ == "__main__":
    evaluate("resultat")
