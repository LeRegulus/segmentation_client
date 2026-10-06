#%% md
# Jour 1 — Préparation et exploration des données (Chapitre 1)

**Projet final « Apprentissage non supervisé » — Master BDA/IA**
Segmentation de clientèle bancaire — jeu *Bank Marketing* (UCI, `bank-additional-full.csv`).

Objectifs du jour :
1. Charger les données et décrire la matrice de données (types de variables).
2. Nettoyage : valeurs `unknown`, doublons, valeurs aberrantes.
3. EDA : statistiques descriptives, histogrammes, scatter plots, heatmap de corrélation.
4. Construire deux représentations : **(a)** one-hot + standardisation $z=(x-\mu)/\sigma$ ; **(b)** données mixtes brutes (Gower / K-prototypes).
5. Réduction de dimension : PCA (variance expliquée), t-SNE et UMAP en 2D.

> La variable cible `y` est **mise de côté** dès le chargement : elle ne sert qu'aux indices externes du Jour 4.
#%% code
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from src.config import SEED, SAMPLE_SIZE, DATA_PROC, FIGURES, MODELS, RESULTS
from src.preprocessing import (load_raw, preprocess, stratified_sample, NUM_FEATURES,
                               BIN_FEATURES, CAT_FEATURES, gower_cat_mask)

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 110
np.random.seed(SEED)
def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
#%% md
## 1. Chargement et description de la matrice de données
#%% code
raw = load_raw()
print("Dimensions :", raw.shape)
raw.head()
#%% code
types = pd.DataFrame({
    "variable": raw.columns,
    "dtype": [str(raw[c].dtype) for c in raw.columns],
    "n_modalites": [raw[c].nunique() for c in raw.columns],
})
nature = {
    "age": "continue (entière)", "duration": "continue (entière)", "campaign": "discrète",
    "pdays": "discrète (999 = jamais)", "previous": "discrète",
    "emp.var.rate": "continue", "cons.price.idx": "continue", "cons.conf.idx": "continue",
    "euribor3m": "continue", "nr.employed": "continue",
    "job": "nominale", "marital": "nominale", "education": "ordinale",
    "default": "binaire (+unknown)", "housing": "binaire (+unknown)", "loan": "binaire (+unknown)",
    "contact": "binaire", "month": "nominale (calendrier)", "day_of_week": "nominale (calendrier)",
    "poutcome": "nominale", "y": "binaire (CIBLE)",
}
types["nature"] = types["variable"].map(nature)
types.to_csv(RESULTS / "j1_types_variables.csv", index=False)
types
#%% md
La matrice de données comporte **41 188 individus × 21 variables** : 10 numériques (dont 5 indicateurs macro-économiques identiques pour tous les clients appelés le même mois) et 11 qualitatives. On distingue continues, discrètes, nominales, une ordinale (`education`) et des binaires — d'où la nécessité de mesures adaptées aux **données mixtes** (Gower, K-prototypes).
#%% md
## 2. Nettoyage : `unknown`, doublons, valeurs aberrantes
#%% code
unk = pd.Series({c: (raw[c] == "unknown").sum() for c in raw.columns if raw[c].dtype == object})
unk = unk[unk > 0].sort_values(ascending=False)
miss = pd.DataFrame({"n_unknown": unk, "%": (100 * unk / len(raw)).round(2)})
print("Doublons exacts :", raw.duplicated().sum())
print("Valeurs NaN     :", raw.isna().sum().sum())
miss
#%% code
num_raw = ["age", "duration", "campaign", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
out = pd.DataFrame({"n_outliers_IQR": {c: iqr_outliers(raw[c]) for c in num_raw}})
out["%"] = (100 * out["n_outliers_IQR"] / len(raw)).round(2)
out["max"] = raw[num_raw].max()
out
#%% code
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
for ax, c in zip(axes, ["age", "duration", "campaign", "previous"]):
    sns.boxplot(x=raw[c], ax=ax, color="#4C72B0"); ax.set_title(c)
fig.suptitle("Boîtes à moustaches — variables à valeurs extrêmes"); fig.tight_layout(); save(fig, "j1_boxplots")
#%% md
**Décisions de nettoyage**

| Problème | Décision | Justification |
|---|---|---|
| 12 doublons exacts | suppression | aucune information supplémentaire |
| `job` (330), `marital` (80) `unknown` | imputation par le mode | < 1 % des lignes |
| `education` (1 731) `unknown` | imputation par le mode **au sein du métier** | 4,2 % ; l'éducation dépend fortement du métier |
| `housing`/`loan` (990, mêmes lignes) | imputation par le mode | 2,4 % |
| `default` : `unknown` = 21 %, `yes` = 3 cas | variable binaire `default_flag` = « non déclaré sans défaut » | `unknown` est informatif (refus de déclarer), `yes` quasi absent |
| `campaign` (max 56, très asymétrique) | écrêtage au 99ᵉ centile (14) puis `log(1+x)` | évite qu'une poignée de clients ne domine les distances euclidiennes |
| `age` (aberrants IQR > 69 ans) | **conservés** | valeurs réelles (retraités), segment métier pertinent |
| `previous` (aberrants IQR dès 1) | **conservés** | 86 % de zéros : tout contact antérieur est « atypique » mais signifiant |
#%% md
## 3. Analyse exploratoire (EDA)
#%% code
raw[num_raw].describe().T.round(2)
#%% code
fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
for ax, c in zip(axes.ravel(), num_raw + ["pdays"]):
    sns.histplot(raw[c], bins=40, ax=ax, color="#4C72B0"); ax.set_title(c); ax.set_xlabel("")
fig.suptitle("Histogrammes des variables numériques"); fig.tight_layout(); save(fig, "j1_histogrammes")
#%% code
cats = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), cats):
    raw[c].value_counts().plot.barh(ax=ax, color="#55A868"); ax.set_title(c); ax.invert_yaxis()
fig.suptitle("Distribution des variables qualitatives"); fig.tight_layout(); save(fig, "j1_barplots_cat")
#%% code
corr = raw[num_raw + ["pdays"]].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
ax.set_title("Heatmap de corrélation (Pearson)"); save(fig, "j1_heatmap_corr")
#%% code
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
s = raw.sample(4000, random_state=SEED)
sns.scatterplot(data=s, x="euribor3m", y="nr.employed", alpha=.3, ax=axes[0], s=12)
sns.scatterplot(data=s, x="emp.var.rate", y="euribor3m", alpha=.3, ax=axes[1], s=12)
sns.scatterplot(data=s, x="age", y="campaign", alpha=.3, ax=axes[2], s=12)
for ax in axes: ax.set_title(f"{ax.get_xlabel()} vs {ax.get_ylabel()}")
fig.tight_layout(); save(fig, "j1_scatter")
#%% code
print("Répartition de la cible (information seulement, non utilisée à l'entraînement) :")
print(raw["y"].value_counts(normalize=True).round(4))
#%% md
**Enseignements de l'EDA**

- Les indicateurs macro-économiques `emp.var.rate`, `euribor3m`, `nr.employed` sont **quasi colinéaires** (r = 0,91 à 0,97) et `cons.price.idx` leur est fortement lié (r ≈ 0,7). Les garder tous reviendrait à pondérer 4 fois la même information dans une distance euclidienne → on conserve **`euribor3m`** (représentant du cycle économique) et **`cons.conf.idx`** (confiance des ménages, peu corrélée aux autres).
- `duration` (durée du dernier appel) n'est connue **qu'après** l'appel : l'utiliser pour segmenter des clients *avant* une campagne serait une **fuite d'information**. Elle est exclue.
- `month` et `day_of_week` décrivent le **calendrier de l'appel**, pas le client : exclues de la segmentation.
- `pdays` vaut 999 (« jamais contacté ») dans 96 % des cas ; l'information est déjà portée par `previous` et `poutcome` : exclue.
- La cible est **déséquilibrée** (11,3 % de souscriptions) : l'échantillon de travail sera **stratifié** sur `y`.
#%% md
## 4. Prétraitement réutilisable et deux représentations

Toute la logique est encapsulée dans `src/preprocessing.py` (fonction `preprocess()`), réutilisée par tous les notebooks et par l'application Streamlit.

Variables de segmentation retenues (13) :
- **numériques / ordinale** (6) : `age`, `campaign_log`, `previous`, `euribor3m`, `cons.conf.idx`, `education_level` (0 = illettré … 6 = université) ;
- **binaires** (4) : `housing`, `loan`, `default_flag`, `contact_cellular` ;
- **nominales** (3) : `job`, `marital`, `poutcome`.
#%% code
data = preprocess()
clean, y, X_a, mixed = data["clean"], data["y"], data["X_a"], data["mixed"]
print(data["report"])
print("Représentation (a) — one-hot + z-score :", X_a.shape)
print("Représentation (b) — mixte brute       :", mixed.shape)
mixed.head()
#%% code
pd.DataFrame(X_a, columns=data["feature_names"]).describe().T[["mean", "std", "min", "max"]].round(2).head(12)
#%% md
Les variables numériques de (a) sont centrées-réduites (moyenne 0, écart-type 1) ; les binaires et indicatrices one-hot restent en 0/1. La représentation (b) garde les variables qualitatives sous forme de modalités (texte) et les numériques dans leur unité d'origine : c'est la distance de **Gower** qui se chargera de la normalisation (par l'étendue).
#%% md
### Échantillon stratifié de travail (n = 4 000)
Les méthodes à matrice de distances (hiérarchique, Gower, K-medoids) ont une complexité mémoire en $O(n^2)$ : on travaille sur un **échantillon aléatoire stratifié** de 4 000 clients, utilisé ensuite par **toutes** les méthodes pour une comparaison équitable.
#%% code
sample_idx = stratified_sample(len(clean), y, SAMPLE_SIZE, SEED)
print("Taille :", len(sample_idx))
print("Taux de 'yes' population :", round(y.mean(), 4), "| échantillon :", round(y.iloc[sample_idx].mean(), 4))
#%% md
## 5. Réduction de dimension
### 5.1 PCA (sur la représentation (a), population complète)
#%% code
pca_full = PCA(random_state=SEED).fit(X_a)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n80, n90 = np.argmax(cum >= .80) + 1, np.argmax(cum >= .90) + 1
print(f"PC1 = {evr[0]:.1%}, PC2 = {evr[1]:.1%}, PC1+PC2 = {cum[1]:.1%}")
print(f"Composantes pour 80 % de variance : {n80} | pour 90 % : {n90}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, len(evr) + 1), evr, label="variance expliquée")
ax.plot(range(1, len(evr) + 1), cum, "o-", color="C3", label="cumulée")
ax.axhline(.8, ls="--", color="grey"); ax.set_xlabel("composante"); ax.legend()
ax.set_title("PCA — variance expliquée (éboulis)"); save(fig, "j1_pca_variance")
#%% code
loadings = pd.DataFrame(pca_full.components_[:3].T, index=data["feature_names"], columns=["PC1", "PC2", "PC3"])
loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(10).round(3)
#%% md
La variance est **diffuse** : aucune composante ne domine (PC1+PC2 ≈ un tiers), ce qui est typique des données mixtes encodées en one-hot. PC1 est portée par le **contexte économique et l'historique de contact** (`euribor3m`, `previous`, `poutcome`, `contact_cellular`), PC2 par le **cycle de vie** (âge, métier retraité/étudiant, statut marital). Pour DBSCAN (Jour 3) on utilisera un espace PCA réduit couvrant ~80 % de la variance ou l'embedding UMAP.
#%% code
Xs = X_a[sample_idx]
pca2 = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
pca_k = PCA(n_components=n80, random_state=SEED).fit_transform(Xs)
tsne2 = TSNE(n_components=2, perplexity=40, init="pca", random_state=SEED).fit_transform(Xs)
umap2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(Xs)
#%% code
color_by = clean.iloc[sample_idx]["poutcome"].values
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, emb, t in zip(axes, [pca2, tsne2, umap2], ["PCA (2 composantes)", "t-SNE (perplexité 40)", "UMAP (30 voisins)"]):
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=color_by, s=8, alpha=.6, ax=ax, linewidth=0)
    ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Projections 2D de l'échantillon (couleur = poutcome, pour lecture uniquement)")
fig.tight_layout(); save(fig, "j1_pca_tsne_umap")
#%% md
**Lecture comparée** : la PCA (linéaire) montre un nuage peu séparé, alors que **t-SNE et UMAP** (non linéaires, préservant la structure locale) font apparaître des **îlots** nets — en particulier les clients déjà contactés (`poutcome` = failure/success) et des sous-groupes liés au contexte économique et au type de contact. UMAP préserve mieux la structure globale que t-SNE et sera réutilisé pour DBSCAN.
#%% md
## 6. Sauvegarde des artefacts pour les jours suivants
#%% code
clean.to_parquet(DATA_PROC / "clean.parquet")
mixed.to_parquet(DATA_PROC / "mixed.parquet")
y.to_frame().to_parquet(DATA_PROC / "y.parquet")
np.save(DATA_PROC / "X_a.npy", X_a)
np.save(DATA_PROC / "sample_idx.npy", sample_idx)
np.save(DATA_PROC / "emb_pca2.npy", pca2); np.save(DATA_PROC / "emb_pca_k.npy", pca_k)
np.save(DATA_PROC / "emb_tsne2.npy", tsne2); np.save(DATA_PROC / "emb_umap2.npy", umap2)
joblib.dump(data["pipeline"], MODELS / "preprocessing_pipeline.joblib")
pd.Series({"pca_n80": n80, "pca_n90": n90, "pc1": evr[0], "pc2": evr[1]}).to_csv(RESULTS / "j1_pca_summary.csv")
print("Artefacts sauvegardés.")
#%% md
**Livrable J1** : ce notebook + la fonction de prétraitement réutilisable `src/preprocessing.py::preprocess()` (et `transform_new_client()` pour affecter un nouveau client).
