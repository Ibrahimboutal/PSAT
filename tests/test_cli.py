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
    # Typer automatically catches negative float boundaries when strictly typed or validated by engine
    # In this case, either Typer stops it or the ValueError inside the engine catches it and we exit(1)!
    assert result.exit_code != 0


