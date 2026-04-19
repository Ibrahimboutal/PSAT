from typer.testing import CliRunner

from psat.cli import app

runner = CliRunner()


def test_cli_default_run():
    # Run the simulation lightly
    result = runner.invoke(app, ["--num-particles", "10", "--time", "0.01", "--visualize", "none"])
    assert result.exit_code == 0
    assert "Simulation Complete" in result.stdout
    assert "Results saved" in result.stdout


def test_cli_invalid_particles():
    result = runner.invoke(app, ["--num-particles", "-50"])
    assert result.exit_code == 1
    assert "Error: Number of particles must be strictly positive" in result.stdout


def test_cli_invalid_time():
    result = runner.invoke(app, ["--time", "-1.0"])
    # Typer returns exit code 2 for parameter validation errors
    assert result.exit_code == 2


def test_cli_visualize_plot(tmp_path, monkeypatch):
    """Cover the visualize=plot branch (lines ~147-154)."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "--num-particles",
            "20",
            "--time",
            "0.01",
            "--visualize",
            "plot",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "trajectories.png" in result.stdout


def test_cli_visualize_animate(tmp_path, monkeypatch):
    """Cover the visualize=animate branch (lines ~136-146)."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "--num-particles",
            "20",
            "--time",
            "0.01",
            "--visualize",
            "animate",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "simulation.gif" in result.stdout


def test_cli_analytics_happy_path(tmp_path, monkeypatch):
    """Cover the --analytics happy path (lines ~168-185) with enough deposits."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "--num-particles",
            "200",
            "--time",
            "0.3",
            "--analytics",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Clustering complete" in result.stdout


def test_cli_analytics_insufficient_deposits(tmp_path, monkeypatch):
    """Cover the --analytics warning branch (line ~174) when too few particles deposit."""
    import unittest.mock as mock

    monkeypatch.chdir(tmp_path)

    # Force all particles to stay active (never deposit) so len(deposited) < 2
    with mock.patch("psat.engine.AerosolSimulation.run", return_value=None):
        result = runner.invoke(
            app,
            [
                "--num-particles",
                "10",
                "--time",
                "0.01",
                "--analytics",
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert "Not enough deposited particles" in result.stdout


def test_cli_cfd_file_happy_path(tmp_path, monkeypatch):
    """Cover the --cfd-file happy path (lines 91-96)."""
    monkeypatch.chdir(tmp_path)

    # 1. Create a dummy CFD CSV
    import pandas as pd

    coords = [(x, y, z) for x in [0.0, 0.1] for y in [-0.01, 0.01] for z in [-0.01, 0.01]]
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["ux"] = 0.5
    df["uy"] = 0.0
    df["uz"] = 0.0
    cfd_path = tmp_path / "flow.csv"
    df.to_csv(cfd_path, index=False)

    # 2. Invoke CLI
    result = runner.invoke(
        app,
        [
            "--num-particles",
            "10",
            "--time",
            "0.01",
            "--cfd-file",
            str(cfd_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "CFD field loaded" in result.stdout


def test_cli_cfd_file_load_failure(tmp_path, monkeypatch):
    """Cover the --cfd-file failure path (lines 98-99)."""
    monkeypatch.chdir(tmp_path)
    # Use a non-existent file
    result = runner.invoke(app, ["--cfd-file", "non_existent.csv"])
    assert result.exit_code == 1, result.stdout
    assert "Failed to load CFD file" in result.stdout
