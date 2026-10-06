# Segmentation de clientèle bancaire par apprentissage non supervisé

Projet final du module **Apprentissage non supervisé** — Master Big Data Analytics / IA (UNCHK).
Auteur : Papa Alassane SECK — octobre 2026.

Jeu de données : *Bank Marketing* (UCI, `bank-additional-full.csv`, 41 188 clients × 21 variables).
La variable cible `y` n'est **jamais** utilisée à l'entraînement : elle sert uniquement aux indices externes (Jour 4) et au calcul *a posteriori* des taux de souscription (Jour 5).

## Résultat principal

Classification mixte **CAH Ward (k = 4) + consolidation K-means** sur 13 variables (silhouette 0,192, Davies-Bouldin 1,69, ARI bootstrap 0,999 à centres fixés).

| Segment | Clients | Souscription | Action |
|---|---|---|---|
| S1 Masse froide (fixe, taux élevés) | 65,4 % | 4,9 % | réduire la pression, canaux à bas coût |
| S2 Jeunes actifs mobiles | 17,0 % | 17,3 % | mobile-first, dépôt court accessible |
| S3 Seniors et retraités épargnants | 6,2 % | 42,5 % | conseiller dédié, dépôt long garanti |
| S4 Clients déjà sollicités | 11,4 % | 22,0 % | renouvellement / nouvel argumentaire |

## Arborescence

```
├── data/
│   ├── bank-additional-full.csv        # données UCI (séparateur ;)
│   ├── online-retail.csv.gz            # second jeu (bonus RFM)
│   └── processed/                      # artefacts intermédiaires (régénérés par le notebook 01, non versionnés)
├── src/
│   ├── config.py                       # graine (42), chemins, taille d'échantillon (4 000), k ∈ [2,10]
│   ├── preprocessing.py                # preprocess(), transform_new_client(), load_processed()
│   ├── metrics.py                      # indice de Dunn (implémentation personnelle), indices internes/externes, coude
│   └── segments.py                     # fiches des 4 segments (profil + recommandation)
├── notebooks/                          # notebooks EXÉCUTÉS (sorties incluses)
│   ├── 01_preparation.ipynb            # Jour 1 — EDA, nettoyage, représentations (a)/(b), PCA, t-SNE, UMAP
│   ├── 02_partitionnement.ipynb        # Jour 2 — distances, standardisation, K-means, K-medoids, K-prototypes, GMM
│   ├── 03_hierarchique_densite.ipynb   # Jour 3 — CAH (4 liens), dendrogrammes, DBSCAN, K-means vs DBSCAN
│   ├── 04_validation.ipynb             # Jour 4 — silhouette, DB, Dunn, ARI, NMI, V-mesure, stabilité, choix final
│   ├── 05_interpretation.ipynb         # Jour 5 — profils, radars, taux de souscription, recommandations
│   ├── 06_bonus_anomalies_selection.ipynb  # Bonus — Isolation Forest, bruit DBSCAN, sélection de variables
│   └── 07_bonus_rfm_online_retail.ipynb    # Bonus — reproduction sur Online Retail (RFM)
├── notebooks_src/                      # sources texte des notebooks (cellules « #%% md » / « #%% code »)
├── app/streamlit_app.py                # Bonus — affectation d'un nouveau client / d'un CSV
├── models/                             # pipeline de prétraitement, K-means final, artefacts de l'application
├── results/                            # tableaux CSV (comparatif J2, synthèse J4, profils J5…)
├── reports/figures/                    # toutes les figures (PNG 150 dpi)
├── rapport/rapport.tex, rapport.pdf    # rapport LaTeX (14 pages)
├── presentation/                       # support de soutenance (.pptx, notes orateur incluses) + script de génération
└── tools/                              # build_notebooks.py (construit + exécute), make_app_assets.py
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

> `scikit-learn-extra` (K-medoids) : la version PyPI 0.3.0 n'est pas compatible avec numpy 2 ; `requirements.txt` installe la version GitHub (nécessite un compilateur C). Alternative : `pip install scikit-learn-extra==0.3.0` dans un environnement avec `numpy<2`.

## Reproduire l'ensemble

```bash
# 1. Construire et exécuter les 7 notebooks dans l'ordre
python tools/build_notebooks.py
# 2. Préparer les artefacts de l'application
python tools/make_app_assets.py
# 3. Lancer l'application
streamlit run app/streamlit_app.py
# 4. Recompiler le rapport
cd rapport && xelatex rapport.tex && xelatex rapport.tex
```

Les notebooks peuvent aussi être ouverts et exécutés un par un dans Jupyter (`notebooks/`), dans l'ordre 01 → 07 : chacun recharge les artefacts produits par les précédents.

## Choix méthodologiques clés

- `duration` exclue (fuite d'information : connue après l'appel) ; `month`/`day_of_week` exclues (calendrier) ; `pdays` exclue (redondante).
- Indicateurs macro colinéaires (r = 0,91–0,97) : seuls `euribor3m` et `cons.conf.idx` sont conservés.
- Deux représentations : (a) one-hot + z-score ; (b) mixte brute pour Gower / K-prototypes.
- Toutes les méthodes sont comparées sur le **même échantillon stratifié de 4 000 clients**, avec des indices internes recalculés dans un espace commun.
- Graine aléatoire fixée à 42 partout (`src/config.py`).
