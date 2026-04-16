import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


def compute_hierarchical_clusters(positions: np.ndarray, max_samples: int = 10000) -> tuple:
    """Isolate coordinates into unsupervised tissue collision group Hot-Spots!

    If length exceeds max_samples, random local down-scaling is implemented
    to preserve RAM limits computationally avoiding Matrix limits.
    """
    n_points = positions.shape[0]
    if n_points == 0:
        return positions, np.array([])

    if n_points > max_samples:
        indices = np.random.choice(n_points, max_samples, replace=False)
        sample_positions = positions[indices]
    else:
        sample_positions = positions.copy()

    # Dynamically find cluster count bounds (distance threshold logic)
    # Using ward linkage minimizing variance of merged groups
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.015,  # 1.5 cm diameter clustering bounds radially
        linkage="ward",
    )
    labels = clustering.fit_predict(sample_positions)

    return sample_positions, labels


def generate_dendrogram(positions: np.ndarray, save_path: str = "cluster_dendrogram.png") -> None:
    """Generate the structural tissue mapping of the Hot-Spot distances."""
    n_points = positions.shape[0]

    # Dendrograms get brutally ugly/messy if over ~500 particles are plotted
    # Downsample aggressively for clean Visualization tree UI
    max_dendrogram_samples = 500

    if n_points > max_dendrogram_samples:
        indices = np.random.choice(n_points, max_dendrogram_samples, replace=False)
        sample_positions = positions[indices]
    else:
        sample_positions = positions

    # Generate Ward Linkage matrix natively
    Z = linkage(sample_positions, "ward")

    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Deposition Tissue Clustering (Dendrogram)")
    plt.xlabel("Particle Sample Index")
    plt.ylabel("Spatial Euclidean Distance (m)")
    dendrogram(Z, truncate_mode="level", p=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
