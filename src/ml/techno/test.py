import os
import pandas as pd
import joblib

# --- Paths ---
base_dir = os.path.dirname(__file__)
path_vectors = os.path.join(base_dir, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
model_path   = os.path.join(base_dir, "..", "..", "..", "models", "lr_multilabel_techno_model.joblib")
out_dir      = os.path.join(base_dir, "..", "..", "..", "output", "predictions")
out_file     = os.path.join(out_dir, "tfidf_vectors_with_tech_predictions.csv")

# --- Paramètres ---
SEUIL_HYBRIDE = 0.4

# --- Keywords ---
SOFT_KEYWORDS = [
    "saas", "software", "platform", "application", "webapp", "cloud", 
    "automation", "analytics", "dashboard", "integration", "management", 
    "solution", "system", "tool", "digital", "iot", "monitoring", "control",
    "predictive", "optimization"
]

HARD_KEYWORDS = [
    "hardware", "sensor", "actuator", "robot", "controller", "circuit", 
    "chip", "fpga", "asic", "board", "module", "device", "component",
    "microcontroller", "motor", "drive", "battery", "inverter", "generator",
    "solar", "wind", "storage", "material", "composite", "nanomaterial"
]

# ---------- Fonctions ----------

def key_predict(row):
    """
    Retourne (label, proba) basé sur les keywords.
    """
    compt_hard = 0
    compt_soft = 0
    for col, val in row.items():
        if col == "doc":
            continue
        if val > 0:
            col_lower = col.lower()
            if any(kw in col_lower for kw in HARD_KEYWORDS):
                compt_hard += 1
            if any(kw in col_lower for kw in SOFT_KEYWORDS):
                compt_soft += 1

    seuil = 0.3
    if compt_hard + compt_soft == 0:
        return "unknown", 0.0
    elif compt_hard > compt_soft * (1 + seuil):
        return "hard", compt_hard / (compt_hard + compt_soft)
    elif compt_soft > compt_hard * (1 + seuil):
        return "soft", compt_soft / (compt_hard + compt_soft)
    else:
        return "both", 0.5


def ml_predict(row, clf, feature_names):
    """
    Retourne (label, proba) basé sur le modèle ML.
    """
    X = pd.DataFrame([row.drop(labels=["doc"])])
    # compléter colonnes manquantes
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0
    X = X[feature_names]

    proba_array = clf.predict_proba(X)

    if isinstance(proba_array, list):  
        hard_prob = proba_array[0][0, 1]
        soft_prob = proba_array[1][0, 1]
    else:  
        hard_prob = proba_array[0, 0]
        soft_prob = proba_array[0, 1]

    if hard_prob >= 0.5 and soft_prob >= 0.5:
        return "both", max(hard_prob, soft_prob)
    elif hard_prob >= 0.5:
        return "hard", hard_prob
    elif soft_prob >= 0.5:
        return "soft", soft_prob
    else:
        return "unknown", max(hard_prob, soft_prob)


def predict_final(path_vectors=path_vectors, out_file=out_file):
    """
    Fusion ML + keywords pour tous les documents.
    """
    clf = joblib.load(model_path)
    df_vec = pd.read_csv(path_vectors, sep=";")

    results = []
    for _, row in df_vec.iterrows():
        doc_name = row["doc"]

        tech_key, prob_key = key_predict(row)
        tech_ml, prob_ml   = ml_predict(row, clf, clf.feature_names_in_)

        prob_ml = prob_ml * (1 + SEUIL_HYBRIDE)

        if prob_ml > prob_key:
            final_label, source = tech_ml, "ml"
        elif prob_key > prob_ml:
            final_label, source = tech_key, "key"
        else:
            final_label, source = tech_ml, "ml/key"

        results.append({
            "doc": doc_name,
            "predicted_tech": final_label,
            "source": source
        })

    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_results.to_csv(out_file, sep=";", index=False)
    print(f"Résultats sauvegardés : {out_file}")

    return df_results


# --- Execution ---
if __name__ == "__main__":
    df_final = predict_final()
