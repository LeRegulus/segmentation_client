#%% md
# Jour 4 — Validation des clusters (Chapitre 4)

Objectifs :
1. **Indices internes** : silhouette, Davies-Bouldin, Dunn — toutes les méthodes, $k\in[2,10]$.
2. **Indices externes** par rapport à `y` (souscription) : ARI, NMI, homogénéité, complétude, V-mesure.
3. **Stabilité** : K-means relancé avec 20 graines + rééchantillonnage bootstrap, ARI entre exécutions.
4. **Sélection** de la meilleure solution en justifiant le compromis entre indices.

> Rappel : `y` n'a été utilisée par **aucun** algorithme. Elle intervient ici uniquement comme vérité terrain externe.
#%% code
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import SEED, K_RANGE, DATA_PROC, FIGURES, RESULTS, MODELS
from src.preprocessing import load_processed
from src.metrics import dunn_index, external_indices, internal_indices, pairwise_ari

sns.set_theme(style="whitegrid", palette="tab10"); plt.rcParams["figure.dpi"] = 110
def save(fig, name): fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)

d = load_processed()
idx = d["sample_idx"]; X_full = d["X_a"]; Xs = X_full[idx]
y_s = d["y"].iloc[idx].to_numpy(); y_full = d["y"].to_numpy()
D_euc = squareform(pdist(Xs)); D_gow = np.load(DATA_PROC / "D_gower_sample.npy")
L2 = joblib.load(RESULTS / "j2_labels_par_k.joblib"); L3 = joblib.load(RESULTS / "j3_labels.joblib")
#%% md
## 1. Indices internes et externes pour toutes les méthodes et tous les $k$
Pour comparer des méthodes reposant sur des géométries différentes, **tous les indices internes sont calculés dans le même espace** : la représentation (a) de l'échantillon (n = 4 000). La silhouette sur Gower est ajoutée pour les méthodes « mixtes ».
#%% code
families = {
    "K-means": L2["kmeans"], "K-medoids (Gower)": L2["kmedoids"], "K-prototypes": L2["kprototypes"],
    "GMM": L2["gmm"],
    "CAH Ward": {k: L3["hac"][("euclidienne", "ward", k)] for k in K_RANGE},
    "CAH average (Gower)": {k: L3["hac"][("gower", "average", k)] for k in K_RANGE},
}
rows = []
for name, labs in families.items():
    for k, lab in labs.items():
        lab = np.asarray(lab)
        ii = internal_indices(Xs, lab)
        ii["dunn"] = dunn_index(lab, D=D_euc)
        sil_g = silhouette_score(D_gow, lab, metric="precomputed")
        sizes = np.bincount(lab)
        rows.append({"méthode": name, "k": k, **ii, "silhouette_gower": sil_g,
                     "min_cluster_%": 100 * sizes.min() / len(lab), **external_indices(y_s, lab)})
for name, lab in [("DBSCAN (UMAP)", L3["dbscan_umap"]), ("DBSCAN (PCA)", L3["dbscan_pca"])]:
    ii = internal_indices(Xs, lab); ii["dunn"] = dunn_index(lab, D=D_euc)
    m = lab != -1
    sizes = np.bincount(lab[m])
    rows.append({"méthode": name, **ii, "silhouette_gower": silhouette_score(D_gow[np.ix_(m, m)], lab[m], metric="precomputed"),
                 "min_cluster_%": 100 * sizes.min() / len(lab), **external_indices(y_s, lab)})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "j4_indices_complets.csv", index=False)
res.round(3).head(12)
#%% code
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
plot_df = res[~res["méthode"].str.startswith("DBSCAN")]
for ax, (col, t) in zip(axes.ravel(), [("silhouette", "Silhouette (a) ↑"), ("davies_bouldin", "Davies-Bouldin ↓"),
                                       ("dunn", "Dunn ↑"), ("ARI", "ARI vs y ↑"), ("NMI", "NMI vs y ↑"),
                                       ("V_mesure", "V-mesure vs y ↑")]):
    sns.lineplot(data=plot_df, x="k", y=col, hue="méthode", marker="o", ax=ax, legend=(col == "silhouette"))
    ax.set_title(t)
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); save(fig, "j4_indices_par_k")
#%% md
**Lecture des indices internes**
- La silhouette décroît globalement avec $k$ : la structure la plus marquée est grossière (2-3 groupes). Les méthodes euclidiennes (K-means, K-prototypes, Ward) dominent dans l'espace (a).
- L'indice de **Dunn** est très faible partout (< 0,1) : il dépend d'un seul couple de points (min / max) et est très sensible au moindre client atypique — attendu sur 4 000 points avec des groupes qui se touchent.
- **Davies-Bouldin** est le plus discriminant entre méthodes : K-means et K-prototypes restent les plus bas.

**Lecture des indices externes**
- ARI et NMI sont **faibles** (souvent < 0,1) : c'est **normal et même souhaitable**. Une segmentation de clientèle ne cherche pas à reproduire la décision de souscription (binaire, déséquilibrée) mais à décrire des profils ; un ARI élevé signifierait que l'on a simplement retrouvé `y`.
- L'**homogénéité** (chaque cluster contient-il une seule classe ?) est plus parlante que la complétude : on cherche des segments où le **taux de souscription varie fortement** (segments « chauds » et « froids »), ce que confirmera le profilage du Jour 5.
#%% code
summary = (res.sort_values("silhouette", ascending=False)
           .groupby("méthode").head(1)[["méthode", "k", "silhouette", "davies_bouldin", "dunn", "silhouette_gower",
                                        "min_cluster_%", "ARI", "NMI", "homogeneite", "completude", "V_mesure"]])
summary.round(3)
#%% md
## 2. Étude de stabilité de K-means
### 2.1 Sensibilité à l'initialisation : 20 graines
On relance K-means **avec une seule initialisation** (`n_init=1`) pour 20 graines différentes et on mesure l'ARI entre toutes les paires d'exécutions (190 paires), pour chaque $k$.
#%% code
seeds = list(range(20))
stab_rows = []
for k in K_RANGE:
    labs = [KMeans(k, n_init=1, random_state=s).fit_predict(X_full) for s in seeds]
    a = pairwise_ari(labs)
    stab_rows.append({"k": k, "ARI_graines_moy": a.mean(), "ARI_graines_min": a.min()})
seed_df = pd.DataFrame(stab_rows); seed_df.round(3)
#%% md
### 2.2 Rééchantillonnage bootstrap
Pour chaque $k$ : un modèle de référence sur toutes les données ; 20 échantillons bootstrap (tirage avec remise, taille $n$) ; chaque modèle bootstrap affecte **toute la population** et on mesure l'ARI avec la référence.
#%% code
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(X_full), len(X_full)) for _ in range(20)]
for row in stab_rows:
    k = row["k"]
    ref = KMeans(k, n_init=10, random_state=SEED).fit(X_full).labels_
    aris = [adjusted_rand_score(ref, KMeans(k, n_init=3, random_state=SEED + b).fit(X_full[bi]).predict(X_full))
            for b, bi in enumerate(boot_idx)]
    row["ARI_bootstrap_moy"], row["ARI_bootstrap_std"] = np.mean(aris), np.std(aris)
stab = pd.DataFrame(stab_rows)
stab.to_csv(RESULTS / "j4_stabilite.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(stab["k"], stab["ARI_graines_moy"], "o-", label="20 graines (n_init=1)")
ax.errorbar(stab["k"], stab["ARI_bootstrap_moy"], yerr=stab["ARI_bootstrap_std"], fmt="s--", capsize=3, label="bootstrap (20)")
ax.set_xlabel("k"); ax.set_ylabel("ARI moyen entre exécutions"); ax.set_ylim(0, 1.05); ax.legend()
ax.set_title("Stabilité de K-means selon k"); save(fig, "j4_stabilite")
stab.round(3)
#%% md
## 3. Sélection de la meilleure solution
Critères de décision (par ordre d'importance) :
1. **Actionnabilité** : 3 à 6 segments (le brief demande 3-4 recommandations), aucun segment < 5 % de la population ;
2. **Qualité interne** : silhouette ↑ et Davies-Bouldin ↓ dans l'espace commun ;
3. **Stabilité** : ARI bootstrap élevé ;
4. **Pouvoir discriminant métier** : homogénéité / V-mesure vis-à-vis de `y` (sans chercher à maximiser l'ARI) ;
5. **Déployabilité** : possibilité d'affecter un nouveau client (exclut CAH et DBSCAN).

On calcule un **score composite par rangs** (moyenne des rangs sur silhouette, Davies-Bouldin, V-mesure) parmi les solutions admissibles.
#%% code
cand = res[(res["k"].between(3, 6)) & (res["min_cluster_%"] >= 5)].copy()
cand["rang_sil"] = cand["silhouette"].rank(ascending=False)
cand["rang_db"] = cand["davies_bouldin"].rank(ascending=True)
cand["rang_v"] = cand["V_mesure"].rank(ascending=False)
cand["score_rangs"] = cand[["rang_sil", "rang_db", "rang_v"]].mean(axis=1)
cand = cand.sort_values("score_rangs")
cand.to_csv(RESULTS / "j4_classement.csv", index=False)
cand[["méthode", "k", "silhouette", "davies_bouldin", "dunn", "V_mesure", "homogeneite", "ARI",
      "min_cluster_%", "score_rangs"]].head(10).round(3)
#%% md
**Résultat du classement** : la **CAH Ward à k = 4** arrive première sur les trois critères à la fois (meilleure silhouette admissible, plus faible Davies-Bouldin, meilleure V-mesure) avec un plus petit segment de 8 %. Mais la CAH ne sait pas **affecter un nouveau client** et ne passe pas à l'échelle des 41 000 clients ($O(n^2)$).

➡️ Solution retenue : **classification mixte** (approche classique en analyse des données) — la CAH Ward fixe le nombre de classes et fournit des centres initiaux, puis une **consolidation par K-means** sur la population complète (initialisée avec les 4 barycentres Ward). On obtient un modèle déployable (centroïdes) qui hérite de la qualité de Ward.
#%% code
k_best = 4
ward4 = families["CAH Ward"][k_best]
centres_ward = np.vstack([Xs[ward4 == c].mean(axis=0) for c in range(k_best)])
final = KMeans(n_clusters=k_best, init=centres_ward, n_init=1, random_state=SEED).fit(X_full)
labels_full = final.labels_
lab_s = labels_full[idx]
print("Accord Ward ↔ consolidation (ARI sur l'échantillon) :", round(adjusted_rand_score(ward4, lab_s), 3))
ii = internal_indices(Xs, lab_s); ii["dunn"] = dunn_index(lab_s, D=D_euc)
cons_row = {"méthode": "Ward + K-means (retenue)", "k": k_best, **ii,
            "silhouette_gower": silhouette_score(D_gow, lab_s, metric="precomputed"),
            "min_cluster_%": 100 * np.bincount(labels_full).min() / len(labels_full), **external_indices(y_s, lab_s)}
pd.Series(cons_row).to_frame("valeur")
#%% code
# Stabilité de la solution mixte : bootstrap de toute la chaîne (CAH Ward sur l'échantillon → K-means)
from scipy.cluster.hierarchy import fcluster, linkage
rng = np.random.default_rng(SEED); aris = []
for b in range(20):
    bi = rng.choice(len(Xs), len(Xs)); Xb = Xs[bi]
    wb = fcluster(linkage(Xb, "ward"), k_best, "maxclust") - 1
    Cb = np.vstack([Xb[wb == c].mean(axis=0) for c in range(k_best)])
    aris.append(adjusted_rand_score(labels_full, KMeans(k_best, init=Cb, n_init=1).fit(X_full).labels_))
aris = np.array(aris)
print(f"ARI bootstrap de la chaîne complète : {aris.mean():.3f} ± {aris.std():.3f} | médiane {np.median(aris):.3f}")
print(f"Part des réplications avec ARI > 0,9 : {(aris > 0.9).mean():.0%}")
cons_row["ARI_bootstrap"] = aris.mean()
#%% code
# Stabilité de la partition finale vis-à-vis de l'échantillonnage (centres Ward fixés, 20 bootstraps de la population)
aris_fin = []
for b in range(20):
    bi = rng.integers(0, len(X_full), len(X_full))
    aris_fin.append(adjusted_rand_score(labels_full, KMeans(k_best, init=final.cluster_centers_, n_init=1).fit(X_full[bi]).predict(X_full)))
aris_fin = np.array(aris_fin)
km4 = KMeans(k_best, n_init=10, random_state=SEED).fit(X_full)
print(f"ARI bootstrap (consolidation, init fixée) : {aris_fin.mean():.4f} ± {aris_fin.std():.4f}")
print(f"Inertie : consolidation Ward→K-means = {final.inertia_:,.0f} | K-means k=4 standard = {km4.inertia_:,.0f}")
print(f"ARI entre les deux solutions k=4 : {adjusted_rand_score(labels_full, km4.labels_):.3f}")
#%% md
**Lecture de la stabilité**
- **K-means seul** : stable en bootstrap pour k = 2, 4, 5 (ARI ≥ 0,94), plus fragile à k = 3 et k ≥ 9 ; la sensibilité à l'initialisation (20 graines, `n_init=1`) est réelle (ARI moyen 0,6-0,9) → d'où `n_init ≥ 10` en pratique.
- **Solution retenue** : une fois les centres Ward fixés, la partition est **extrêmement stable** au rééchantillonnage (ARI ≈ 0,999). L'instabilité vient de la **chaîne complète** (ARI 0,72 ± 0,23) : sur certains bootstraps, la CAH coupe différemment les petits groupes (seniors / déjà contactés). C'est une limite à documenter ; elle plaide pour **figer** les centres de référence et les ré-estimer périodiquement.
- La consolidation a une inertie un peu plus élevée que K-means standard (optimum local différent) mais une **meilleure silhouette (0,192 vs 0,148) et un meilleur Davies-Bouldin (1,69 vs 1,94)** : elle sépare mieux les groupes au lieu de seulement réduire la variance intra.
- Indices externes : ARI = 0,157, homogénéité = 0,125 — les meilleurs parmi les solutions admissibles, et surtout des taux de souscription très contrastés (4,9 % → 42,5 %), analysés au Jour 5.

#%% code
# Tableau de synthèse final (livrable J4) : meilleure solution admissible de chaque famille + DBSCAN + solution retenue
synth = pd.concat([cand.groupby("méthode").head(1), res[res["méthode"].str.startswith("DBSCAN")],
                   pd.DataFrame([cons_row])])
synth = synth[["méthode", "k", "silhouette", "davies_bouldin", "dunn", "silhouette_gower", "ARI", "NMI",
               "homogeneite", "completude", "V_mesure", "min_cluster_%"]].reset_index(drop=True)
synth.to_csv(RESULTS / "j4_synthese.csv", index=False)
synth.round(3)
#%% md
## 4. Modèle final sur la population complète
#%% code
joblib.dump(final, MODELS / "kmeans_final.joblib")
np.save(DATA_PROC / "labels_final.npy", labels_full)
pd.DataFrame({"ARI_chaine": aris, "ARI_consolidation": aris_fin}).to_csv(RESULTS / "j4_stabilite_solution_finale.csv", index=False)
print("Tailles des segments :", np.bincount(labels_full))
print("Indices externes (population complète) :", {k: round(v, 4) for k, v in external_indices(y_full, labels_full).items()})
print("Taux de souscription par segment :", pd.Series(y_full).groupby(labels_full).mean().round(3).to_dict())
