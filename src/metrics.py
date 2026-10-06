"""Indices de validation interne / externe (Chapitre 4)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import (adjusted_rand_score, completeness_score, davies_bouldin_score,
                             homogeneity_score, normalized_mutual_info_score, silhouette_score,
                             v_measure_score)


def dunn_index(labels: np.ndarray, X: np.ndarray | None = None, D: np.ndarray | None = None) -> float:
    """Indice de Dunn = min distance inter-clusters / max diamètre intra-cluster.

    Implémentation personnelle. On fournit soit X (espace euclidien), soit une
    matrice de dissimilarité D (n × n), par exemple la matrice de Gower.
    Les points de bruit (label -1, DBSCAN) sont ignorés.
    """
    labels = np.asarray(labels)
    mask = labels != -1
    labels = labels[mask]
    if D is None:
        D = squareform(pdist(X[mask], metric="euclidean"))
    else:
        D = D[np.ix_(mask, mask)]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    groups = [np.where(labels == k)[0] for k in uniq]
    max_diam = max(D[np.ix_(g, g)].max() if len(g) > 1 else 0.0 for g in groups)
    min_sep = np.inf
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            min_sep = min(min_sep, D[np.ix_(groups[i], groups[j])].min())
    return float(min_sep / max_diam) if max_diam > 0 else np.nan


def internal_indices(X: np.ndarray, labels: np.ndarray, D: np.ndarray | None = None) -> dict:
    """Silhouette, Davies-Bouldin, Dunn dans l'espace X (bruit DBSCAN exclu).

    Si D est fourni (matrice précalculée), la silhouette « native » sur D est aussi calculée.
    """
    labels = np.asarray(labels)
    m = labels != -1
    out = {"k": int(len(np.unique(labels[m]))), "noise_%": float(100 * (~m).mean())}
    if out["k"] < 2:
        return {**out, "silhouette": np.nan, "davies_bouldin": np.nan, "dunn": np.nan}
    out["silhouette"] = float(silhouette_score(X[m], labels[m]))
    out["davies_bouldin"] = float(davies_bouldin_score(X[m], labels[m]))
    out["dunn"] = dunn_index(labels, X=X)
    if D is not None:
        out["silhouette_native"] = float(silhouette_score(D[np.ix_(m, m)], labels[m], metric="precomputed"))
    return out


def external_indices(y_true: np.ndarray, labels: np.ndarray) -> dict:
    """ARI, NMI, homogénéité, complétude, V-mesure par rapport à la vérité terrain y."""
    return {
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "homogeneite": homogeneity_score(y_true, labels),
        "completude": completeness_score(y_true, labels),
        "V_mesure": v_measure_score(y_true, labels),
    }


def pairwise_ari(label_sets: list[np.ndarray]) -> np.ndarray:
    """ARI pour toutes les paires d'exécutions (étude de stabilité)."""
    n = len(label_sets)
    return np.array([adjusted_rand_score(label_sets[i], label_sets[j])
                     for i in range(n) for j in range(i + 1, n)])


def knee_point(values: np.ndarray) -> int:
    """Indice du « coude » d'une courbe croissante (distance max à la corde)."""
    v = np.asarray(values, dtype=float)
    x = np.arange(len(v))
    p1, p2 = np.array([0, v[0]]), np.array([len(v) - 1, v[-1]])
    a, b = p2 - p1, np.c_[x, v] - p1
    d = np.abs(a[0] * b[:, 1] - a[1] * b[:, 0]) / np.linalg.norm(a)
    return int(np.argmax(d))


def elbow_k(ks, inertias) -> int:
    """Coude d'une courbe décroissante d'inertie (distance max à la corde normalisée)."""
    ks, w = np.asarray(list(ks), float), np.asarray(inertias, float)
    xs = (ks - ks.min()) / (ks.max() - ks.min())
    ws = (w - w.min()) / (w.max() - w.min())
    d = np.abs(xs + ws - 1) / np.sqrt(2)   # distance à la droite (0,1)-(1,0)
    return int(ks[np.argmax(d)])
