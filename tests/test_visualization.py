import os

import numpy as np

from psat.visualization import plot_deposition, plot_trajectories, plot_trajectories_plotly


def test_plot_trajectories_save(tmp_path):
    trajectories = np.random.rand(5, 10, 3)  # 5 steps, 10 particles, 3D
    domain = ((0, 1), (0, 1), (0, 1))

    save_file = tmp_path / "traj.png"
    plot_trajectories(trajectories, domain, num_particles_to_plot=5, save_path=str(save_file))

    assert os.path.exists(save_file)


def test_plot_deposition_save(tmp_path):
    final_positions = np.random.rand(10, 3)
    wall_dep = np.array([True] * 5 + [False] * 5)
    bot_dep = np.array([False] * 5 + [True] * 5)
    domain = ((0, 1), (0, 1), (0, 1))

    save_file = tmp_path / "dep.png"
    plot_deposition(
        final_positions,
        domain,
        wall_deposit=wall_dep,
        bottom_deposit=bot_dep,
        save_path=str(save_file),
    )

    assert os.path.exists(save_file)


def test_plotly_generation():
    trajectories = np.random.rand(2, 5, 3)
    domain = ((0, 1), (0, 1), (0, 1))

    fig = plot_trajectories_plotly(trajectories, domain, num_particles_to_plot=2)
    assert fig is not None
    assert "xaxis" in fig.layout.scene


def test_animate_trajectories_save(tmp_path):
    import os

    from psat.visualization import animate_trajectories

    trajectories = np.random.rand(2, 5, 3)  # 2 steps, 5 particles, 3D
    domain = ((0, 1), (0, 1), (0, 1))

    save_file = tmp_path / "sim.gif"

    # Animate rendering utilizing native Pillow backend
    animate_trajectories(
        trajectories, domain, num_particles_to_plot=2, save_path=str(save_file), fps=1
    )

    assert os.path.exists(save_file)
