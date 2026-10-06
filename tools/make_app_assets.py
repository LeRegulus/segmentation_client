"""Prépare les artefacts légers de l'application Streamlit (à lancer après les notebooks 01→04)."""
import sys; sys.path.insert(0, ".")
import joblib, numpy as np
from sklearn.decomposition import PCA
from src.config import MODELS, SEED
from src.preprocessing import load_processed

d = load_processed()
X, idx = d["X_a"], d["sample_idx"]
labels = np.load("data/processed/labels_final.npy")
km = joblib.load(MODELS / "kmeans_final.joblib")
pca = PCA(2, random_state=SEED).fit(X)
np.savez_compressed(MODELS / "app_background.npz", xy=pca.transform(X[idx]), labels=labels[idx],
                    centers=pca.transform(km.cluster_centers_))
joblib.dump(pca, MODELS / "pca2_app.joblib")
print("OK")
