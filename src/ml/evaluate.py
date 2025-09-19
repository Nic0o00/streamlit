"""
Évaluation simple des prédictions pour un label (ex: tech, domain, country).
Compare predicted_xxx avec le label réel et calcule les bons, faux et le pourcentage.
Affiche également la liste des documents mal prédits avec leur prédiction et label réel.
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate(label_col):
    """
    Compare les prédictions label_col avec les labels du fichier labeled_total.csv.
    Exemple : evaluate("tech"), evaluate("domain"), evaluate("country")
    """
    # --- Chemins ---
    pred_file = f"tfidf_vectors_with_{label_col}_predictions.csv"
    predict_col = f"predicted_{label_col}"
    base_dir = os.path.dirname(__file__)
    path_pred  = os.path.join(base_dir, "..", "..", "output", "predictions", pred_file)
    path_label = os.path.join(base_dir, "..", "..", "labeled_total.csv")

    print(f"\n=== Évaluation pour {label_col} ===")
    print(f"Fichier prédictions : {path_pred}")
    print(f"Fichier labels      : {path_label}")

    # --- Lecture ---
    df_pred = pd.read_csv(path_pred, sep=";")
    df_label = pd.read_csv(path_label, sep=";")

    # --- Harmonisation ---
    df_pred["doc"] = df_pred["doc"].astype(str)
    df_label["doc"] = df_label["doc"].astype(str)

    # --- Jointure sur "doc" ---
    df_eval = pd.merge(
        df_pred[["doc", predict_col]],
        df_label[["doc", label_col]],
        on="doc", how="inner"
    )

    if len(df_eval) == 0:
        raise ValueError(f"Aucune correspondance entre prédictions et labels pour {label_col}.")

    # --- Comparaison ---
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
    labels_sorted = sorted(df_eval[label_col].unique())
    cm = confusion_matrix(df_eval[label_col], df_eval[predict_col], labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print("\n--- Matrice de confusion ---")
    print(cm_df)

    # Retourne les résultats pour analyse externe
    return df_eval, df_wrong, cm_df


if __name__ == "__main__":
    # Exemple d’utilisation
    evaluate("tech")
    evaluate("domain")
    evaluate("country")
