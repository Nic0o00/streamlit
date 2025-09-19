from functions import (
    load_txt_files, build_snippets_df, load_tech_predictions, 
    build_label_embeddings, heuristic_boost, combine_labels_weighted,
    REVENUE_MODELS, HEURISTICS, CATEGORY_TO_REVENUE,
    CHUNK_CHARS, CHUNK_OVERLAP, MODEL_NAME
)
from sentence_transformers import SentenceTransformer
import joblib
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
INPUT_DIR = os.path.join(base_dir,"..","..","..","data", "processed","translated")
TECH_CSV = os.path.join(base_dir,"..","..","..","output", "predictions","tfidf_vectors_with_tech_predictions.csv")
MODELS_DIR = os.path.join(base_dir,"..","..","..","models")


def predict_pipeline(input_dir=INPUT_DIR, out_dir=os.path.join(base_dir,"..","..","..","output","predictions")):
    os.makedirs(out_dir, exist_ok=True)

    #  Charger les modèles
    clf = joblib.load(os.path.join(MODELS_DIR, "clf_model.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    
    #  Charger et découper les documents
    docs = load_txt_files(input_dir)
    df_snips = build_snippets_df(docs, CHUNK_CHARS, CHUNK_OVERLAP)

    #  Fusion avec les prédictions tech
    tech_df = load_tech_predictions(TECH_CSV)
    df_snips = df_snips.merge(tech_df, on="doc", how="left")

    #  Créer la colonne base_label à partir de CATEGORY_TO_REVENUE
    df_snips["base_label"] = df_snips["predicted_label"].map(CATEGORY_TO_REVENUE)

    #  Créer les embeddings et les embeddings de labels
    model = SentenceTransformer(MODEL_NAME)
    label_names, label_emb = build_label_embeddings(model, REVENUE_MODELS)
    embeddings = model.encode(df_snips["snippet"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

    #  Calcul des scores embeddings
    scores = embeddings @ label_emb.T

    #  Heuristic boost
    boosted_scores = []
    for snippet, base in zip(df_snips["snippet"], scores):
        boosted_scores.append(heuristic_boost(snippet, label_names, base, HEURISTICS))
    boosted_scores = pd.DataFrame(boosted_scores, columns=label_names)

    #  Prédiction du classifieur
    preds_idx = clf.predict(embeddings)
    preds_embeddings = le.inverse_transform(preds_idx)

    #  Fusion base_label + embeddings avec pondération 75%-25%
    df_snips["pred_label"] = [
        combine_labels_weighted(base, emb, row)
        for base, emb, row in zip(
            df_snips["base_label"],
            preds_embeddings,
            boosted_scores.to_dict("records")
        )
    ]

    #  Calcul des confidences du classifieur
    probs = clf.predict_proba(embeddings)
    confidences = probs.max(axis=1)

    #  Création du dataframe final par snippet
    df_final = pd.DataFrame({
        "doc": df_snips["doc"],
        "pred_label": df_snips["pred_label"],
        "confidence": confidences
    })

    #  Agrégation par document
    df_doc = df_final.groupby("doc").apply(
        lambda g: pd.Series({
            "pred_label": g.loc[g["confidence"].idxmax(), "pred_label"],
            "confidence": g["confidence"].max()
        })
    ).reset_index()

    # 1Sauvegarde
    out_path = os.path.join(out_dir, "predictions_revenu.csv")
    df_doc.to_csv(out_path, index=False)
    print(f"Prediction completed. Results saved to {out_path}")
    return df_doc


if __name__ == "__main__":
    predict_pipeline()
