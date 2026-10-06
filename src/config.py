"""Configuration globale du projet (chemins, graine, tailles)."""
from pathlib import Path

SEED = 42                      # graine aléatoire fixée partout (reproductibilité)
SAMPLE_SIZE = 4000             # échantillon stratifié pour les méthodes à matrice de distances
K_RANGE = range(2, 11)         # k ∈ [2, 10]

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "bank-additional-full.csv"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

for _p in (DATA_PROC, RESULTS, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
