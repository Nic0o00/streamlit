from functions import (load_txt_files, build_snippets_df, load_tech_predictions, 
                        build_label_embeddings, heuristic_boost, combine_labels, REVENUE_MODELS, HEURISTICS, CATEGORY_TO_REVENUE,
                        CHUNK_CHARS, CHUNK_OVERLAP, MODEL_NAME)
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import joblib

# -------------------------
# Paths
# -------------------------
base_dir = os.path.dirname(__file__)
INPUT_DIR = os.path.join(base_dir,"..","..","..","data", "processed","translated")
TECH_CSV = os.path.join(base_dir,"..","..","..","output", "predictions","tfidf_vectors_with_tech_predictions.csv")
MODELS_DIR = os.path.join(base_dir,"..","..","..","models")


# -------------------------
# Pipeline d'entraînement
# -------------------------
def train_pipeline():
    docs = load_txt_files(INPUT_DIR)
    df_snips = build_snippets_df(docs, CHUNK_CHARS, CHUNK_OVERLAP)
    
    tech_df = load_tech_predictions(TECH_CSV)
    df_snips = df_snips.merge(tech_df, on="doc", how="left")

    # mapping catégorie hard/soft/both -> revenu
    df_snips["base_label"] = df_snips["predicted_label"].map(CATEGORY_TO_REVENUE)

    # embeddings
    model = SentenceTransformer(MODEL_NAME)
    label_names, label_emb = build_label_embeddings(model, REVENUE_MODELS)
    embeddings = model.encode(df_snips["snippet"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
    scores = embeddings @ label_emb.T  

    # heuristic boost
    boosted_scores = []
    for snippet, base in zip(df_snips["snippet"], scores):
        boosted_scores.append(heuristic_boost(snippet, label_names, base, HEURISTICS))
    boosted_scores = pd.DataFrame(boosted_scores, columns=label_names)

    # prédictions embeddings
    preds_embeddings = boosted_scores.idxmax(axis=1)


    df_snips["pred_label"] = [
        combine_labels(base, emb, row) 
        for base, emb, row in zip(df_snips["base_label"], preds_embeddings, boosted_scores.to_dict("records"))
    ]

    # entraînement classifieur
    le = LabelEncoder()
    y = le.fit_transform(df_snips["pred_label"])
    clf = LogisticRegression(max_iter=300)
    clf.fit(embeddings, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "clf_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    print("✅ Training completed. Model saved to", MODELS_DIR)
    return clf, le


if __name__ == "__main__":
    train_pipeline()
