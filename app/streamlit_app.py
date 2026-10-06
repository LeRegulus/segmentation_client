"""Application Streamlit — affecter un nouveau client bancaire à un segment.

Lancement (depuis la racine du dépôt) :
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import EDU_ORDER, transform_new_client
from src.segments import SEG_COLORS, SEGMENTS

st.set_page_config(page_title="Segmentation clients — Bank Marketing", page_icon="🏦", layout="wide")


@st.cache_resource
def load_models():
    pipe = joblib.load(ROOT / "models" / "preprocessing_pipeline.joblib")
    km = joblib.load(ROOT / "models" / "kmeans_final.joblib")
    pca = joblib.load(ROOT / "models" / "pca2_app.joblib")
    bg = np.load(ROOT / "models" / "app_background.npz")
    return pipe, km, pca, bg


pipe, km, pca, bg = load_models()

JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
        "services", "student", "technician", "unemployed"]


def predict(client: dict):
    x = transform_new_client(client, pipe)
    dist = np.linalg.norm(km.cluster_centers_ - x, axis=1)
    seg = int(np.argmin(dist))
    w = np.exp(-dist); w /= w.sum()          # affinité relative (softmax des distances, indicatif)
    return seg, dist, w, x


st.title("🏦 Segmentation de la clientèle bancaire")
st.caption("Projet final — Apprentissage non supervisé (Master BDA/IA) · modèle : CAH Ward + consolidation K-means, k = 4")

tab1, tab2, tab3 = st.tabs(["👤 Nouveau client", "📄 Affectation par lot (CSV)", "📊 Les 4 segments"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profil")
        age = st.slider("Âge", 17, 98, 40)
        job = st.selectbox("Profession", JOBS, index=0)
        marital = st.selectbox("Statut marital", ["married", "single", "divorced"])
        education = st.selectbox("Niveau d'études", list(EDU_ORDER), index=6)
    with c2:
        st.subheader("Situation bancaire")
        housing = st.radio("Prêt immobilier", ["yes", "no"], horizontal=True)
        loan = st.radio("Prêt personnel", ["no", "yes"], horizontal=True)
        default = st.radio("Défaut de crédit déclaré", ["no", "unknown", "yes"], horizontal=True)
        contact = st.radio("Canal de contact", ["cellular", "telephone"], horizontal=True)
    with c3:
        st.subheader("Historique & contexte")
        campaign = st.number_input("Nb d'appels prévus dans la campagne", 1, 56, 1)
        previous = st.number_input("Nb de contacts lors de campagnes passées", 0, 7, 0)
        poutcome = st.selectbox("Résultat de la campagne précédente", ["nonexistent", "failure", "success"])
        euribor = st.slider("Euribor 3 mois (%)", 0.6, 5.1, 1.3, 0.05)
        conf = st.slider("Indice de confiance des ménages", -51.0, -26.0, -42.0, 0.5)

    if previous == 0 and poutcome != "nonexistent":
        st.warning("Incohérence : un résultat de campagne précédente implique au moins 1 contact antérieur.")

    client = dict(age=age, job=job, marital=marital, education=education, housing=housing, loan=loan,
                  default=default, contact=contact, campaign=campaign, previous=previous, poutcome=poutcome,
                  euribor3m=euribor, **{"cons.conf.idx": conf})
    seg, dist, w, x = predict(client)
    s = SEGMENTS[seg]

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"<h3 style='color:{SEG_COLORS[seg]}'>Segment S{seg+1} · {s['nom']}</h3>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Taux de souscription historique", f"{s['taux']:.1f} %")
        m2.metric("Lift vs moyenne (11,3 %)", f"×{s['lift']:.2f}")
        st.markdown(f"**Profil type** — {s['profil']}")
        st.success(f"**Action recommandée** — {s['reco']}")
        st.markdown("**Distance aux centres des segments**")
        st.dataframe(pd.DataFrame({"segment": [f"S{k+1} · {SEGMENTS[k]['court']}" for k in range(4)],
                                   "distance": dist.round(3), "affinité relative": (100 * w).round(1)}),
                     hide_index=True, width="stretch")
    with right:
        fig, ax = plt.subplots(figsize=(6, 5))
        for k in range(4):
            m = bg["labels"] == k
            ax.scatter(bg["xy"][m, 0], bg["xy"][m, 1], s=4, alpha=.35, c=SEG_COLORS[k], label=f"S{k+1}")
            ax.scatter(*bg["centers"][k], marker="X", s=180, c=SEG_COLORS[k], edgecolor="k")
        p = pca.transform(x)[0]
        ax.scatter(*p, marker="*", s=500, c="gold", edgecolor="k", label="nouveau client", zorder=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8, loc="best")
        ax.set_title("Position du client (projection PCA)")
        st.pyplot(fig)

with tab2:
    st.markdown("Chargez un CSV au format `bank-additional-full.csv` (séparateur `;`). Les colonnes inutiles "
                "(`duration`, `month`, `y`…) sont ignorées.")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up, sep=";")
        need = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "campaign",
                "previous", "poutcome", "euribor3m", "cons.conf.idx"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            d = df[need].copy()
            for c in ["job", "marital", "housing", "loan"]:          # mêmes imputations que l'entraînement
                d[c] = d[c].replace("unknown", {"job": "admin.", "marital": "married",
                                                "housing": "yes", "loan": "no"}[c])
            d["education"] = d["education"].replace("unknown", "university.degree")
            segs = [predict(r)[0] for r in d.to_dict("records")]
            df["segment"] = [f"S{k+1} · {SEGMENTS[k]['court']}" for k in segs]
            st.bar_chart(df["segment"].value_counts())
            st.dataframe(df.head(200), width="stretch")
            st.download_button("Télécharger les affectations", df.to_csv(index=False, sep=";").encode(),
                               "clients_segmentes.csv", "text/csv")

with tab3:
    for k, s in SEGMENTS.items():
        st.markdown(f"<h4 style='color:{SEG_COLORS[k]}'>S{k+1} · {s['nom']} — souscription {s['taux']} % "
                    f"(lift ×{s['lift']})</h4>", unsafe_allow_html=True)
        st.markdown(f"{s['profil']}\n\n➡️ **{s['reco']}**")
