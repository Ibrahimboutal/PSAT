import numpy as np

from psat.analytics import compute_hierarchical_clusters, generate_dendrogram


def _make_two_blobs() -> np.ndarray:
    """Synthetic data: two tight 3D spherical blobs well separated in space."""
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[0.02, 0.0, 0.0], scale=0.001, size=(100, 3))
    blob_b = rng.normal(loc=[0.08, 0.0, 0.0], scale=0.001, size=(100, 3))
    return np.vstack([blob_a, blob_b])


def test_clustering_returns_correct_shapes():
    """Cluster function must return same-length positions and labels arrays."""
    positions = _make_two_blobs()
    sample_pos, labels = compute_hierarchical_clusters(positions)
    assert sample_pos.shape[1] == 3, "Positions must have 3 columns (x, y, z)"
    assert len(labels) == len(sample_pos), "Label count must match sample count"


def test_clustering_finds_two_blobs():
    """Two spatially separated blobs should resolve into exactly 2 clusters."""
    positions = _make_two_blobs()
    _, labels = compute_hierarchical_clusters(positions)
    n_clusters = len(np.unique(labels))
    assert n_clusters == 2, f"Expected 2 clusters for two blobs, got {n_clusters}"


def test_clustering_downsamples_large_input():
    """Down-sampling logic must cap samples at max_samples without error."""
    rng = np.random.default_rng(0)
    big_positions = rng.random((15000, 3))
    sample_pos, labels = compute_hierarchical_clusters(big_positions, max_samples=500)
    assert len(sample_pos) == 500
    assert len(labels) == 500


def test_clustering_empty_input():
    """Empty input should return empty arrays gracefully."""
    empty = np.empty((0, 3))
    sample_pos, labels = compute_hierarchical_clusters(empty)
    assert len(sample_pos) == 0
    assert len(labels) == 0


def test_generate_dendrogram_saves_file(tmp_path):
    """Dendrogram function must save a PNG to the given path."""
    import os

    positions = _make_two_blobs()
    out = tmp_path / "dendro.png"
    generate_dendrogram(positions, save_path=str(out))
    assert os.path.exists(out), "Dendrogram PNG was not created"
    assert os.path.getsize(out) > 0, "Dendrogram PNG is empty"


def test_plot_clusters_plotly_returns_figure():
    """Plotly cluster function must return a valid Figure with one trace per cluster."""
    from psat.visualization import plot_deposition_clusters_plotly

    positions = _make_two_blobs()
    _, labels = compute_hierarchical_clusters(positions)
    domain = ((0.0, 0.1), (-0.01, 0.01), (-0.01, 0.01))
    fig = plot_deposition_clusters_plotly(positions, labels, domain)
    assert fig is not None
    n_clusters = len(np.unique(labels))
    assert len(fig.data) == n_clusters, "One trace per cluster expected"
