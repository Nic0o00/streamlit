# predict_domain_select.py
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
MODELS_DIR = os.path.join(BASE, "..", "..", "..", "models")
OUT = os.path.join(BASE, "..", "..", "..", "output", "predictions")
os.makedirs(OUT, exist_ok=True)
OUT_FILE = os.path.join(OUT, "tfidf_vectors_with_domain_predictions.csv")

def predict_domain():

    selector = joblib.load(os.path.join(MODELS_DIR, "domain_selector.joblib"))
    svd = joblib.load(os.path.join(MODELS_DIR, "domain_svd.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "domain_scaler.joblib"))
    centroids = joblib.load(os.path.join(MODELS_DIR, "domain_centroids.joblib"))
    clf = joblib.load(os.path.join(MODELS_DIR, "domain_clf.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "domain_label_encoder.joblib"))
    nonzero_cols = joblib.load(os.path.join(MODELS_DIR, "domain_nonzero_columns.joblib"))

    df_vec = pd.read_csv(VECT_CSV, sep=";")
    df_vec["doc"] = df_vec["doc"].astype(str)
    docs = df_vec["doc"].tolist()

    # ensure nonzero_cols exist
    for c in nonzero_cols:
        if c not in df_vec.columns:
            df_vec[c] = 0
    X_df = df_vec[nonzero_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_sp = csr_matrix(X_df.values)

    # select -> sparse
    X_sel = selector.transform(X_sp)  # sparse
    # cosine similarity to centroids
    cos_sim = cosine_similarity(X_sel, centroids)  # dense
    # svd + scale
    X_red = svd.transform(X_sel)
    X_scaled = scaler.transform(X_red)

    X_final = np.hstack([X_scaled, cos_sim])

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_final)
        pred_idx = np.argmax(proba, axis=1)
        maxp = proba.max(axis=1)
    else:
        pred_idx = clf.predict(X_final)
        maxp = np.ones(len(pred_idx))

    pred_lab = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"doc": docs, "predicted_domain": pred_lab, "confidence_score": maxp})
    out.to_csv(OUT_FILE, sep=";", index=False)
    print("Predictions saved to", OUT_FILE)

if __name__ == "__main__":
    predict_domain()
