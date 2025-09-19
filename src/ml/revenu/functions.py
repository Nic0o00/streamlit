import os
import pandas as pd
import numpy as np
from sentence_transformers import util

# -------------------------
# Global dictionaries
# -------------------------
REVENUE_MODELS = {
    "subscription": (
        "Subscription-based revenue: recurring payments for access to a digital product or service. "
        "Examples: SaaS platforms, cloud services, IoT monitoring platforms, predictive maintenance subscriptions, "
        "AI analytics tools, or digital twins. Typically billed monthly or annually (MRR/ARR)."
    ),
    "one_time": (
        "One-time revenue: single purchase or sale of physical products, machinery, industrial robots, "
        "equipment, or advanced materials. Includes installation, commissioning, and professional services "
        "linked to the delivery. Usually considered CAPEX or one-off project billing."
    ),
    "recurring": (
        "Recurring revenue from physical products or services tied to repeat use: consumables, spare parts, "
        "maintenance contracts, warranties, managed services, or energy supply agreements. Payments are "
        "periodic, usage-based, or linked to ongoing client needs beyond subscriptions."
    ),
}

HEURISTICS = {
    "subscription": [
        "subscription", "saas", "platform", "cloud", "license fee", "user license",
        "monthly fee", "annual fee", "mrr", "arr", "per seat", "digital twin",
        "predictive maintenance", "iot monitoring", "analytics", "software access", "software"
    ],
    "one_time": [
        "one-time", "single payment", "purchase", "equipment", "machinery", "robotics",
        "installation", "setup fee", "commissioning", "hardware", "advanced material",
        "capital expenditure", "capex", "industrial machine", "sale", "project delivery"
    ],
    "recurring": [
        "recurring", "recurrent", "recurrence", "maintenance contract", "service agreement",
        "spare parts", "consumables", "renewal", "support contract", "annual contract",
        "energy service", "managed service", "usage-based", "pay per use", "leasing",
        "outsourcing", "long-term agreement", "maintenance fee", "long term", "selling H2"
        "selling energy"
    ],
}

CATEGORY_TO_REVENUE = {
    "hard": "one_time",
    "soft": "subscription",
}

# Marge relative : fraction du score de base que l'embedding doit dépasser pour renverser.
MARGIN_REL = 0.05
# Poids appliqué à la base (category -> revenue)
BASE_WEIGHT = 0.75
CHUNK_CHARS = 300
CHUNK_OVERLAP = 50
BATCH_SIZE = 16
MIN_CONF = 0.4
MIN_MARGIN = 0.1
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------
# I/O & preprocessing
# -------------------------
def load_txt_files(input_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                full = os.path.join(root, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    doc_base = os.path.splitext(fn)[0]
                    files.append((doc_base, text))
                except Exception:
                    continue
    return files


def build_snippets_df(docs, chunk_chars, chunk_overlap):
    records = []
    for doc_base, text in docs:
        text_norm = " ".join(text.split())
        if not text_norm:
            continue
        if len(text_norm) <= chunk_chars:
            chunks = [text_norm]
        else:
            chunks = []
            start = 0
            step = max(1, chunk_chars - chunk_overlap)
            while start < len(text_norm):
                end = min(start + chunk_chars, len(text_norm))
                chunks.append(text_norm[start:end])
                if end == len(text_norm):
                    break
                start += step
        for i, ch in enumerate(chunks):
            records.append({"doc": doc_base, "snippet_id": f"{doc_base}_{i}", "snippet": ch})
    return pd.DataFrame(records)


def load_tech_predictions(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tech CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    df = df.drop(columns="confidence_score")
    df = df.rename(columns={df.columns[0]: "doc", df.columns[1]: "predicted_label"})
    df["doc"] = df["doc"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    return df


# -------------------------
# Label embeddings & scoring
# -------------------------
def build_label_embeddings(model, revenue_models):
    label_names = list(revenue_models.keys())
    label_texts = [revenue_models[k] for k in label_names]
    label_emb = model.encode(label_texts, convert_to_numpy=True, normalize_embeddings=True)
    return label_names, label_emb


def heuristic_boost(snippet, label_names, base_scores, heuristics):
    s = snippet.lower()
    boost = np.zeros_like(base_scores)
    for i, label in enumerate(label_names):
        for kw in heuristics.get(label, []):
            if kw in s:
                boost[i] += 0.03
    return base_scores + boost

# fusion avec base_label
def combine_labels(base, emb_pred, scores_row):
    # si pas de base (NaN)
    if base is None or pd.isna(base):
        return emb_pred  

    # si la base existe bien dans les scores_row
    if base in scores_row:
        if scores_row[emb_pred] > scores_row[base] + 0.2:  
            return emb_pred
        else:
            return base
    else:
        return emb_pred

# Nouvelle fonction pour fusion base + embedding
def combine_labels_weighted(base, emb_pred, scores_row, alpha=None, margin_rel=None):
    """
    Combine base_label et emb_pred en pondérant :
      - alpha (poids de la base). Si None, on utilise BASE_WEIGHT.
      - margin_rel (marge relative). Si None, on utilise MARGIN_REL.

    Retourne le label final (base ou emb_pred).
    """
    # utiliser valeurs par défaut si non fournies
    if alpha is None:
        alpha = BASE_WEIGHT
    if margin_rel is None:
        margin_rel = MARGIN_REL

    # cas pas de base -> suivre embedding
    if base is None or pd.isna(base):
        return emb_pred

    # sécurité : si un des labels absent dans scores_row -> suivre embedding (ou base selon choix)
    if (base not in scores_row) or (emb_pred not in scores_row):
        return emb_pred

    base_score = float(scores_row[base])
    emb_score = float(scores_row[emb_pred])

    weighted_base = alpha * base_score
    weighted_emb = (1.0 - alpha) * emb_score

    # margin_value = margin_rel * base_score  (scale-invariant)
    margin_value = margin_rel * base_score

    if weighted_emb > (weighted_base + margin_value):
        return emb_pred
    else:
        return base