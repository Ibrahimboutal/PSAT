"""Tests for Phase 5 — Time-dependent CFD loading and engine integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psat.cfd_loader import TimeDependentFlowField, detect_and_load, load_csv_flow, wrap_steady_flow
from psat.engine import AerosolSimulation


def _write_time_varying_csv(path: Path, n_grid: int = 4) -> None:
    """axial flow grows linearly with time: ux = 0.5 at t=0, ux = 1.0 at t=1."""
    data = []
    # Grid of points
    xs = np.linspace(0, 0.1, n_grid)
    ys = np.linspace(-0.01, 0.01, n_grid)
    zs = np.linspace(-0.01, 0.01, n_grid)

    for t in [0.0, 1.0]:
        ux_val = 0.5 if t == 0.0 else 1.0
        for x in xs:
            for y in ys:
                for z in zs:
                    data.append(
                        {"x": x, "y": y, "z": z, "ux": ux_val, "uy": 0.0, "uz": 0.0, "t": t}
                    )

    pd.DataFrame(data).to_csv(path, index=False)


def test_time_dependent_interpolation(tmp_path):
    csv = tmp_path / "dynamic_flow.csv"
    _write_time_varying_csv(csv)

    flow = load_csv_flow(csv)
    assert isinstance(flow, TimeDependentFlowField)

    # Query at t=0.5 (should be (0.5 + 1.0)/2 = 0.75)
    x = np.array([0.05])
    y = np.array([0.0])
    z = np.array([0.0])
    ux, uy, uz = flow(x, y, z, t=0.5)

    assert np.isclose(ux[0], 0.75, atol=1e-3)
    assert np.isclose(uy[0], 0.0)


def test_time_clamping(tmp_path):
    csv = tmp_path / "dynamic_flow.csv"
    _write_time_varying_csv(csv)
    flow = load_csv_flow(csv)

    x = np.zeros(1)
    # Beyond t=1.0 should stay at 1.0
    ux, _, _ = flow(x, x, x, t=2.0)
    assert np.isclose(ux[0], 1.0)

    # Before t=0.0 should stay at 0.5
    ux, _, _ = flow(x, x, x, t=-0.5)
    assert np.isclose(ux[0], 0.5)


def test_wrap_steady_flow():
    def dummy_3arg(x, y, z):
        return x, y, z

    wrapped = wrap_steady_flow(dummy_3arg)

    # Should accept 4 args now
    ux, uy, uz = wrapped(np.array([1.0]), np.array([2.0]), np.array([3.0]), 100.0)
    assert ux[0] == 1.0
    assert uy[0] == 2.0
    assert uz[0] == 3.0


def test_engine_with_dynamic_flow(tmp_path):
    csv = tmp_path / "dynamic_flow.csv"
    _write_time_varying_csv(csv)
    flow = load_csv_flow(csv)

    # Run a small simulation
    sim = AerosolSimulation(
        num_particles=5,
        dt=0.01,
        total_time=0.1,
        domain_limits=((0, 0.1), (-0.01, 0.01), (-0.01, 0.01)),
        fluid_velocity_func=flow,
    )
    sim.initialize_particles()
    sim.run()
    # If it didn't crash, the signature call was successful
    assert sim.positions.shape == (5, 3)


def test_detect_and_load_2d_time_csv(tmp_path):
    csv = tmp_path / "flow_2d.csv"
    data = [
        {"x": 0.0, "y": 0.0, "u": 1.0, "v": 0.0, "t": 0.0},
        {"x": 0.1, "y": 0.0, "u": 1.0, "v": 0.0, "t": 0.0},
        {"x": 0.0, "y": 0.1, "u": 1.0, "v": 0.0, "t": 0.0},
        {"x": 0.1, "y": 0.1, "u": 1.0, "v": 0.0, "t": 0.0},
        {"x": 0.0, "y": 0.0, "u": 2.0, "v": 0.0, "t": 1.0},
        {"x": 0.1, "y": 0.0, "u": 2.0, "v": 0.0, "t": 1.0},
        {"x": 0.0, "y": 0.1, "u": 2.0, "v": 0.0, "t": 1.0},
        {"x": 0.1, "y": 0.1, "u": 2.0, "v": 0.0, "t": 1.0},
    ]
    pd.DataFrame(data).to_csv(csv, index=False)

    flow = detect_and_load(csv)
    ux, _, _ = flow(np.array([0.05]), np.array([0.05]), np.array([0.0]), t=0.5)
    assert np.isclose(ux[0], 1.5)
