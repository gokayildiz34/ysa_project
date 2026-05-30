"""
Klasik modellerin mimari/model tanımı — sadece model oluşturma fonksiyonları.
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42


def build_isolation_forest():
    """Isolation Forest modelini oluşturur."""
    return IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_ocsvm():
    """One-Class SVM modelini oluşturur."""
    return OneClassSVM(
        kernel="rbf",
        nu=0.05,
        gamma="scale",
    )


def build_pca():
    """PCA Reconstruction modelini oluşturur."""
    return PCA(n_components=0.95, random_state=RANDOM_STATE)
