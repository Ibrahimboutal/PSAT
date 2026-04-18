"""
PSAT Visualization Module
==========================
Provides static trajectory plots, deposition histograms, and animated GIFs
for inspecting aerosol simulation results.
"""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_trajectories(
    trajectories: np.ndarray,
    domain_limits: tuple,
    num_particles_to_plot: int = 100,
    save_path: str | None = None,
) -> None:
    """Render a static 3-D overlay of particle trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Full trajectory history with shape ``(n_steps, N, 3)``.
    domain_limits : tuple
        Spatial bounds ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    num_particles_to_plot : int, optional
        Maximum number of randomly sampled particles to display. Default 100.
    save_path : str, optional
        File path to save the figure (PNG/PDF).  If *None*, displays
        interactively.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_particles: int = trajectories.shape[1]
    plot_idx: np.ndarray = np.random.choice(
        n_particles, min(n_particles, num_particles_to_plot), replace=False
    )

    for idx in plot_idx:
        x = trajectories[:, idx, 0]
        y = trajectories[:, idx, 1]
        z = trajectories[:, idx, 2]
        ax.plot(x, y, z, alpha=0.5, linewidth=0.8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("Axial Position x (m)")
    ax.set_ylabel("Radial Position y (m)")
    ax.set_zlabel("Depth Position z (m)")
    ax.set_title("Aerosol Particle Trajectories in 3D Airway")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_deposition(
    final_positions: np.ndarray,
    domain_limits: tuple,
    wall_deposit: np.ndarray | None = None,
    bottom_deposit: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a histogram of particle deposition positions along the airway axis.

    Parameters
    ----------
    final_positions : np.ndarray
        Array of shape ``(N, 3)`` with each particle's last recorded position.
    domain_limits : tuple
        Spatial bounds ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    wall_deposit : np.ndarray of bool, optional
        Boolean mask of shape ``(N,)`` flagging wall-deposited particles.
    bottom_deposit : np.ndarray of bool, optional
        Boolean mask of shape ``(N,)`` flagging outlet-deposited particles.
        Currently reserved for future use.
    save_path : str, optional
        File path to save the figure.  If *None*, displays interactively.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    plt.figure(figsize=(10, 4))

    if wall_deposit is not None:
        wall_positions = final_positions[wall_deposit]
        plt.hist(
            wall_positions[:, 0],
            bins=40,
            range=(xmin, xmax),
            alpha=0.7,
            color="crimson",
            edgecolor="black",
            label="Wall Deposition",
        )

    # Legacy fallback when wall_deposit mask is not available
    if wall_deposit is None:
        floor_deposited = np.isclose(final_positions[:, 1], ymin, atol=1e-3)
        plt.hist(
            final_positions[floor_deposited, 0],
            bins=50,
            range=(xmin, xmax),
            alpha=0.7,
            color="blue",
            edgecolor="black",
            label="Floor Deposition",
        )

    plt.xlabel("Axial Position along Airway x (m)")
    plt.ylabel("Number of Deposited Particles")
    plt.title("Wall Deposition Distribution Profile")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def animate_trajectories(
    trajectories: np.ndarray,
    domain_limits: tuple,
    num_particles_to_plot: int = 100,
    save_path: str = "simulation.gif",
    fps: int = 30,
) -> None:
    """Generate an animated GIF of particle motion through the airway.

    The animation is down-sampled to at most 150 frames to keep file sizes
    manageable while preserving visual fidelity.

    Parameters
    ----------
    trajectories : np.ndarray
        Full trajectory history with shape ``(n_steps, N, 3)``.
    domain_limits : tuple
        Spatial bounds ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    num_particles_to_plot : int, optional
        Maximum number of randomly sampled particles to animate. Default 100.
    save_path : str, optional
        Output file path.  Use ``.gif`` extension for Pillow writer, or
        ``.mp4`` for ffmpeg. Default ``"simulation.gif"``.
    fps : int, optional
        Frames per second of the output animation. Default 30.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_steps: int
    n_particles: int
    n_steps, n_particles, _ = trajectories.shape
    plot_idx: np.ndarray = np.random.choice(
        n_particles, min(n_particles, num_particles_to_plot), replace=False
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("Axial Position x (m)")
    ax.set_ylabel("Radial Position y (m)")
    ax.set_zlabel("Depth Position z (m)")
    ax.set_title("Aerosol Particle Trajectories in 3D Airway")

    scat = ax.scatter([], [], [], s=2, alpha=0.5, color="blue")

    def update(frame: int):  # type: ignore[return]
        x = trajectories[frame, plot_idx, 0]
        y = trajectories[frame, plot_idx, 1]
        z = trajectories[frame, plot_idx, 2]
        scat._offsets3d = (x, y, z)
        return (scat,)

    # Limit to ~150 frames to control output size
    step: int = max(1, n_steps // 150)
    frames = range(0, n_steps, step)

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=1000 / fps)

    if save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=fps)
    else:
        ani.save(save_path, fps=fps)

    plt.close()


def plot_trajectories_plotly(
    trajectories: np.ndarray,
    domain_limits: tuple,
    num_particles_to_plot: int = 100,
) -> go.Figure:
    """Generate an interactive Plotly 3D scatter plot of aerosol trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        Full trajectory history with shape ``(n_steps, N, 3)``.
    domain_limits : tuple
        Spatial bounds ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    num_particles_to_plot : int, optional
        Maximum number of randomly sampled particles to display. Default 100.

    Returns
    -------
    go.Figure
        A Plotly Figure object that can be rendered in a Streamlit app.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    n_particles = trajectories.shape[1]
    plot_idx = np.random.choice(n_particles, min(n_particles, num_particles_to_plot), replace=False)

    fig = go.Figure()

    # Add trajectories
    for idx in plot_idx:
        x = trajectories[:, idx, 0]
        y = trajectories[:, idx, 1]
        z = trajectories[:, idx, 2]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=2, color="rgba(79, 142, 247, 0.5)"),
                showlegend=False,
            )
        )

    # Add invisible boundary boxes to force Plotly to scale correctly
    fig.add_trace(
        go.Scatter3d(
            x=[xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin],
            y=[ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax],
            z=[zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],
            mode="markers",
            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Configure Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Axial x (m)",
            yaxis_title="Radial y (m)",
            zaxis_title="Depth z (m)",
            xaxis=dict(range=[xmin, xmax], gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(range=[ymin, ymax], gridcolor="rgba(255,255,255,0.2)"),
            zaxis=dict(range=[zmin, zmax], gridcolor="rgba(255,255,255,0.2)"),
            bgcolor="rgba(14, 17, 23, 1)",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor="rgba(14, 17, 23, 1)",
        font=dict(color="white"),
    )

    return fig


def plot_deposition_clusters_plotly(
    positions: np.ndarray,
    labels: np.ndarray,
    domain_limits: tuple,
):
    """Render Hot-Spot groups identified via unsupervised hierarchical clustering.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Deposited particle coordinates (m).
    labels : np.ndarray, shape (N,)
        Integer cluster label for every sampled particle.
    domain_limits : tuple
        ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    """
    import plotly.graph_objects as go

    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits

    fig = go.Figure()
    unique_labels = np.unique(labels)
    palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    for label in unique_labels:
        idx = labels == label
        pts = positions[idx]
        color = palette[int(label) % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.8),
                name=f"Cluster {label}",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="Axial x (m)",
            yaxis_title="Radial y (m)",
            zaxis_title="Depth z (m)",
            xaxis=dict(range=[xmin, xmax], gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(range=[ymin, ymax], gridcolor="rgba(255,255,255,0.2)"),
            zaxis=dict(range=[zmin, zmax], gridcolor="rgba(255,255,255,0.2)"),
            bgcolor="rgba(14, 17, 23, 1)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(14, 17, 23, 1)",
        font=dict(color="white"),
        title="Hierarchical Tissue Clustering — Hot-Spots",
    )
    return fig
