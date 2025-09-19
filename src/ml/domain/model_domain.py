# train_domain_select.py
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(__file__)
VECT_CSV = os.path.join(BASE, "..", "..", "..", "data", "processed", "tfidf_vectors.csv")
LAB_CSV  = os.path.join(BASE, "..", "..", "..", "data", "labeled.csv")
MODELS_DIR = os.path.join(BASE, "..", "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# hyperparams
K_SELECT = 3000        # nombre de features à garder après sélection chi2
SVD_COMPONENTS = 150   # réduction après sélection
CLASS_WEIGHTED = True

def _canon_label(s):
    return str(s).strip().lower() if pd.notna(s) else "unknown"

def train_domain():
    # --- chargement ---
    df_vec = pd.read_csv(VECT_CSV, sep=";")
    df_lab = pd.read_csv(LAB_CSV, sep=";")
    df_vec["doc"] = df_vec["doc"].astype(str)
    df_lab["doc"] = df_lab["doc"].astype(str)

    # --- merge + labels canonisés ---
    df = pd.merge(df_vec, df_lab, on="doc", how="inner")
    if df.shape[0] == 0:
        raise RuntimeError("Aucune correspondance entre vecteurs et labels")
    df["domain_y"] = df["domain_y"].apply(_canon_label)

    # --- features ---
    X_df = df.drop(columns=["doc","domain_y"], errors="ignore")
    # conversion numérique
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # drop colonnes totalement nulles
    col_sums = X_df.sum(axis=0)
    nonzero_cols = col_sums[col_sums > 0].index.tolist()
    X_df = X_df[nonzero_cols]
    print("Features after zero-drop:", X_df.shape[1])

    # --- labels ---
    y_raw = df["domain_y"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # --- sparse ---
    X_sp = csr_matrix(X_df.values)

    # --- feature selection ---
    k = min(K_SELECT, X_sp.shape[1]-1)
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_sp, y)
    X_sel = selector.transform(X_sp)

    # --- centroids ---
    classes = np.unique(y)
    centroids = []
    for c in classes:
        rows = (y == c)
        if rows.sum() == 0:
            centroids.append(np.zeros(X_sel.shape[1], dtype=float))
        else:
            s = X_sel[rows].sum(axis=0)
            centroid = np.asarray(s).ravel() / float(max(1, rows.sum()))
            centroids.append(centroid)
    centroids = np.vstack(centroids)

    # --- cosine similarity ---
    cos_sim = cosine_similarity(X_sel, centroids)

    # --- SVD ---
    n_comp = min(SVD_COMPONENTS, X_sel.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_red = svd.fit_transform(X_sel)

    # --- scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_red)

    # --- features finales ---
    X_final = np.hstack([X_scaled, cos_sim])

    # --- classifier ---
    clf = LogisticRegression(class_weight="balanced" if CLASS_WEIGHTED else None,
                             solver="saga", max_iter=2000)
    clf.fit(X_final, y)

    # --- sauvegarde ---
    joblib.dump(selector, os.path.join(MODELS_DIR, "domain_selector.joblib"))
    joblib.dump(svd, os.path.join(MODELS_DIR, "domain_svd.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "domain_scaler.joblib"))
    joblib.dump(centroids, os.path.join(MODELS_DIR, "domain_centroids.joblib"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "domain_clf.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "domain_label_encoder.joblib"))
    joblib.dump(nonzero_cols, os.path.join(MODELS_DIR, "domain_nonzero_columns.joblib"))

    print("Training complete. Saved in", MODELS_DIR)

if __name__ == "__main__":
    train_domain()
