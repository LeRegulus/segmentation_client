"""
Prétraitement réutilisable – Segmentation de clientèle bancaire (Bank Marketing, UCI).

Point d'entrée principal : ``preprocess()`` qui renvoie un dictionnaire contenant
- ``clean``    : DataFrame nettoyé (variables de segmentation uniquement) ;
- ``y``        : vérité terrain (0/1) — JAMAIS utilisée pour l'entraînement ;
- ``X_a``      : représentation (a) one-hot + standardisation (np.ndarray dense) ;
- ``mixed``    : représentation (b) données mixtes brutes (Gower / K-prototypes) ;
- ``pipeline`` : ColumnTransformer ajusté (réutilisable pour un nouveau client).

Choix de modélisation (justifiés dans le notebook 01 et le rapport) :
- ``duration`` est EXCLUE : elle n'est connue qu'après l'appel (fuite d'information) ;
- ``month`` / ``day_of_week`` sont exclues : elles décrivent le calendrier de l'appel,
  pas le client ;
- ``pdays`` (999 = jamais contacté, 96 % des lignes) est redondante avec ``previous`` /
  ``poutcome`` : exclue ;
- ``emp.var.rate``, ``nr.employed`` et ``cons.price.idx`` sont très corrélées à
  ``euribor3m`` (r = 0.97 / 0.95 / 0.69) : on garde ``euribor3m`` + ``cons.conf.idx`` ;
- ``campaign`` est très asymétrique : écrêtage au 99e centile puis log(1 + x).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_RAW, SAMPLE_SIZE, SEED

# --------------------------------------------------------------------------- #
# Définition des variables de segmentation
# --------------------------------------------------------------------------- #
EDU_ORDER = {
    "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
    "high.school": 4, "professional.course": 5, "university.degree": 6,
}
NUM_FEATURES = ["age", "campaign_log", "previous", "euribor3m", "cons.conf.idx", "education_level"]
BIN_FEATURES = ["housing", "loan", "default_flag", "contact_cellular"]
CAT_FEATURES = ["job", "marital", "poutcome"]
SEG_FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES

CAMPAIGN_CAP = 14  # 99e centile de `campaign` sur le jeu complet


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    """Charge le CSV brut (séparateur ';')."""
    df = pd.read_csv(path, sep=";")
    # pandas >= 3 : colonnes texte en dtype 'str' → on repasse en object (compatibilité gower/kmodes)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object)
    return df


def clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Nettoyage + ingénierie de variables. Renvoie (clean, y, rapport)."""
    report = {"n_raw": len(df_raw)}
    df = df_raw.drop_duplicates().reset_index(drop=True)
    report["n_duplicates"] = len(df_raw) - len(df)

    y = (df["y"] == "yes").astype(int).rename("y")
    report["unknown_counts"] = {c: int((df[c] == "unknown").sum())
                                for c in df.columns if df[c].dtype == object}

    out = pd.DataFrame(index=df.index)
    # ---- numériques
    out["age"] = df["age"].astype(float)
    camp = df["campaign"].clip(upper=CAMPAIGN_CAP)
    report["campaign_capped"] = int((df["campaign"] > CAMPAIGN_CAP).sum())
    out["campaign_log"] = np.log1p(camp).astype(float)
    out["previous"] = df["previous"].astype(float)
    out["euribor3m"] = df["euribor3m"].astype(float)
    out["cons.conf.idx"] = df["cons.conf.idx"].astype(float)

    # ---- catégorielles nominales : 'unknown' (< 1 %) imputé par le mode
    for c in ["job", "marital"]:
        mode = df.loc[df[c] != "unknown", c].mode()[0]
        out[c] = df[c].replace("unknown", mode)
    out["poutcome"] = df["poutcome"]

    # ---- ordinale : education (4.2 % unknown) imputée par le mode du métier
    edu = df["education"].where(df["education"] != "unknown")
    edu = edu.fillna(edu.groupby(out["job"]).transform(lambda s: s.mode()[0] if s.notna().any() else np.nan))
    edu = edu.fillna(edu.mode()[0])
    out["education_level"] = edu.map(EDU_ORDER).astype(float)

    # ---- binaires (0/1)
    for c in ["housing", "loan"]:          # unknown (2.4 %, mêmes lignes) → mode
        mode = df.loc[df[c] != "unknown", c].mode()[0]
        out[c] = (df[c].replace("unknown", mode) == "yes").astype(int)
    # default : 'yes' quasi inexistant (3 cas) mais 'unknown' = 21 % → signal « risque non renseigné »
    out["default_flag"] = (df["default"] != "no").astype(int)
    out["contact_cellular"] = (df["contact"] == "cellular").astype(int)

    for c in CAT_FEATURES:
        out[c] = out[c].astype(object)
    report["n_clean"] = len(out)
    return out[SEG_FEATURES], y, report


def make_onehot_pipeline() -> ColumnTransformer:
    """Représentation (a) : z-score sur numériques, one-hot sur nominales, binaires 0/1."""
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_FEATURES),
         ("bin", "passthrough", BIN_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


def to_mixed(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Représentation (b) : données mixtes « brutes » (num. float, qualitatives en str)."""
    m = clean_df.copy()
    for c in BIN_FEATURES:
        m[c] = m[c].map({0: "no", 1: "yes"}).astype(object)
    return m


def gower_cat_mask(mixed: pd.DataFrame) -> list[bool]:
    return [mixed[c].dtype == object for c in mixed.columns]


def stratified_sample(n_total: int, y: pd.Series, n: int = SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
    """Indices d'un échantillon aléatoire stratifié sur y (même proportion de 'yes')."""
    idx = np.arange(n_total)
    sample_idx, _ = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    return np.sort(sample_idx)


def preprocess(path=DATA_RAW) -> dict:
    """Pipeline complet : chargement → nettoyage → représentations (a) et (b)."""
    raw = load_raw(path)
    clean_df, y, report = clean(raw)
    pipe = make_onehot_pipeline()
    X_a = pipe.fit_transform(clean_df)
    return {
        "raw": raw, "clean": clean_df, "y": y, "report": report,
        "pipeline": pipe, "X_a": X_a, "feature_names": list(pipe.get_feature_names_out()),
        "mixed": to_mixed(clean_df),
    }


def transform_new_client(client: dict, pipeline: ColumnTransformer) -> np.ndarray:
    """Transforme un nouveau client (dict de valeurs « métier ») dans l'espace (a).

    Clés attendues : age, campaign, previous, euribor3m, cons.conf.idx, education,
    housing, loan, default, contact, job, marital, poutcome.
    """
    row = {
        "age": float(client["age"]),
        "campaign_log": float(np.log1p(min(client["campaign"], CAMPAIGN_CAP))),
        "previous": float(client["previous"]),
        "euribor3m": float(client["euribor3m"]),
        "cons.conf.idx": float(client["cons.conf.idx"]),
        "education_level": float(EDU_ORDER[client["education"]]),
        "housing": int(client["housing"] == "yes"),
        "loan": int(client["loan"] == "yes"),
        "default_flag": int(client["default"] != "no"),
        "contact_cellular": int(client["contact"] == "cellular"),
        "job": client["job"], "marital": client["marital"], "poutcome": client["poutcome"],
    }
    df = pd.DataFrame([row])[SEG_FEATURES]
    for c in CAT_FEATURES:
        df[c] = df[c].astype(object)
    return pipeline.transform(df)


def load_processed() -> dict:
    """Recharge les artefacts produits par le notebook 01 (évite de tout recalculer)."""
    from .config import DATA_PROC
    def _obj(df):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype(object)
        return df
    d = {
        "clean": _obj(pd.read_parquet(DATA_PROC / "clean.parquet")),
        "mixed": _obj(pd.read_parquet(DATA_PROC / "mixed.parquet")),
        "y": pd.read_parquet(DATA_PROC / "y.parquet")["y"],
        "X_a": np.load(DATA_PROC / "X_a.npy"),
        "sample_idx": np.load(DATA_PROC / "sample_idx.npy"),
    }
    for e in ["pca2", "pca_k", "tsne2", "umap2"]:
        d[f"emb_{e}"] = np.load(DATA_PROC / f"emb_{e}.npy")
    return d
