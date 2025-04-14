import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.
    Returns:
        Pandas DataFrame.
    """
    return pd.read_csv(filepath)


def extract_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Extract numeric feature matrix from DataFrame.

    Args:
        df: Input DataFrame.
        feature_cols: List of column names to use as features.
    Returns:
        2D NumPy array of features.
    """
    return df[feature_cols].to_numpy()


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit KMeans and return labels and centroids.

    Args:
        X: Feature matrix.
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.
    Returns:
        Tuple of (labels array, centers array).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_


def calculate_wcss(
    X: np.ndarray,
    max_clusters: int = 10
) -> List[float]:
    """
    Compute within-cluster sum of squares for 1..max_clusters.

    Args:
        X: Feature matrix.
        max_clusters: Maximum number of clusters to evaluate.
    Returns:
        List of inertia values.
    """
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss