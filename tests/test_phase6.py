import numpy as np
import pytest

from psat.cfd_loader import FlowField, RegularFlowField, load_csv_flow
from psat.engine import AerosolSimulation


def test_hygroscopic_growth_increases_diameter():
    """Verify that particles grow over time when growth_rate > 0."""
    num_p = 10
    dt = 0.01
    rate = 1.0  # 100% per second

    sim = AerosolSimulation(
        num_particles=num_p,
        dt=dt,
        total_time=0.1,
        domain_limits=((0, 1), (-0.1, 0.1), (-0.1, 0.1)),
        mean_diameter=1e-6,
        hygroscopic_growth_rate=rate,
    )

    dp_start = sim.dp.copy()
    sim.initialize_particles()
    sim.step(1)

    # After one step of 0.01s at 100%/s rate, diameter should be ~1% larger
    # d_new = d_old * (1 + 1.0 * 0.01) = d_old * 1.01
    assert np.all(sim.dp > dp_start)
    assert np.allclose(sim.dp, dp_start * (1.0 + rate * dt))


def test_hygroscopic_growth_updates_physics():
    """Verify that Cc and Tau are updated as particles grow."""
    sim = AerosolSimulation(
        num_particles=5,
        dt=0.1,
        total_time=1.0,
        domain_limits=((0, 1), (-1, 1), (-1, 1)),
        hygroscopic_growth_rate=1.0,
    )

    sim.initialize_particles()
    tau_start = sim.tau.copy()

    sim.step(1)
    # Larger particles = higher relaxation time (tau)
    assert np.all(sim.tau > tau_start)


def test_structured_grid_detection(tmp_path):
    """Verify that a regular grid is correctly detected and uses RegularFlowField."""
    import pandas as pd

    # Create a 4x4x4 structured grid
    x = np.linspace(0, 1, 4)
    y = np.linspace(-0.1, 0.1, 4)
    z = np.linspace(-0.1, 0.1, 4)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    df = pd.DataFrame(
        {"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel(), "ux": 0.5, "uy": 0.0, "uz": 0.0}
    )

    csv_path = tmp_path / "structured.csv"
    df.to_csv(csv_path, index=False)

    # Loader should pick RegularFlowField
    field = load_csv_flow(csv_path)
    assert isinstance(field, RegularFlowField)

    # Test query
    ux, uy, uz = field(np.array([0.5]), np.array([0.0]), np.array([0.0]))
    assert ux[0] == pytest.approx(0.5)


def test_unstructured_fallback(tmp_path):
    """Verify that missing points trigger FlowField (LinearND) fallback."""
    import pandas as pd

    # Create a grid but remove one point to make it irregular
    x = np.linspace(0, 1, 3)
    y = np.linspace(-0.1, 0.1, 3)
    z = np.linspace(-0.1, 0.1, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    df = pd.DataFrame(
        {"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel(), "ux": 0.5, "uy": 0.0, "uz": 0.0}
    ).drop(index=5)  # Punch a hole

    csv_path = tmp_path / "unstructured.csv"
    df.to_csv(csv_path, index=False)

    field = load_csv_flow(csv_path)
    assert isinstance(field, FlowField)


def test_cfd_scaling(tmp_path):
    """Verify that the scale parameter correctly transforms CFD units."""
    import pandas as pd

    # Data in mm (0 to 100mm)
    df = pd.DataFrame(
        {"x": [0, 100], "y": [0, 0], "z": [0, 0], "ux": [1, 1], "uy": [0, 0], "uz": [0, 0]}
    )
    csv_path = tmp_path / "scale.csv"
    df.to_csv(csv_path, index=False)

    # Scale=0.001 should convert 100mm to 0.1m
    field = load_csv_flow(csv_path, scale=0.001)

    # Querying at 0.05m (halfway) should work
    ux, _, _ = field(np.array([0.05]), np.array([0.0]), np.array([0.0]))
    assert ux[0] == 1.0
