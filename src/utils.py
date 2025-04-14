import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Sequence
from matplotlib.figure import Figure


def plot_cluster_counts(labels: Sequence[int]) -> Figure:
    """
    Generate a bar chart showing the number of samples in each cluster.

    Args:
        labels: Sequence of integer cluster labels.
    Returns:
        Matplotlib Figure with cluster size distribution.
    """
    # Count and sort cluster sizes
    counts = pd.Series(labels).value_counts().sort_index()

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, edgecolor="black")
    ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster Label", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def visualize_clusters(
    X: np.ndarray,
    labels: Sequence[int],
    centers: np.ndarray
) -> Figure:
    """
    Scatter plot of clustered data with centroids.

    Args:
        X: 2D array of shape (n_samples, 2).
        labels: Cluster labels for each sample.
        centers: 2D array of cluster centroids.
    Returns:
        Matplotlib Figure with clusters and centroids plotted.
    """
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.size

    # Choose a colormap
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, cluster in enumerate(unique_labels):
        mask = labels == cluster
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=50,
            label=f"Cluster {cluster}",
            color=cmap(idx),
            edgecolor='k',
            alpha=0.7
        )

    # Plot centroids
    ax.scatter(
        centers[:, 0], centers[:, 1],
        s=200,
        marker='X',
        c='black',
        label='Centroids',
        linewidths=2
    )

    ax.set_title("Cluster Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel('Annual Income ($K)', fontsize=14)
    ax.set_xlabel('Spending Score', fontsize=14)
    ax.legend(title="Clusters", fontsize=10, title_fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig