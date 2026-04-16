from typer.testing import CliRunner

from psat.cli import app

runner = CliRunner()


def test_optimization_loop():
    # Use just 2 trials and 10 particles so testing is instantaneous natively
    result = runner.invoke(app, ["--optimize", "--trials", "2", "--num-particles", "10"])

    assert result.exit_code == 0
    assert "Optimization Complete" in result.stdout
    assert "Top Therapy Score Found" in result.stdout
