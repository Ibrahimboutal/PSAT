"""Tests for psat/cfd_loader.py — CSV loading, interpolation, error handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psat.cfd_loader import FlowField, detect_and_load, load_csv_flow

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_uniform_csv(path: Path, ux: float = 0.5, n: int = 5) -> None:
    """Write a uniform axial flow field on a small 3-D lattice."""
    coords = [
        (x, y, z)
        for x in np.linspace(0, 0.1, n)
        for y in np.linspace(-0.01, 0.01, n)
        for z in np.linspace(-0.01, 0.01, n)
    ]
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["ux"] = ux
    df["uy"] = 0.0
    df["uz"] = 0.0
    df.to_csv(path, index=False)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_load_csv_returns_flow_field(tmp_path):
    """load_csv_flow must return a FlowField instance."""
    csv = tmp_path / "field.csv"
    _write_uniform_csv(csv)
    flow = load_csv_flow(csv)
    assert isinstance(flow, FlowField)


def test_load_csv_uniform_flow_interior(tmp_path):
    """Interior query of a uniform 0.5 m/s axial field must return ≈ 0.5."""
    csv = tmp_path / "field.csv"
    _write_uniform_csv(csv, ux=0.5, n=6)
    flow = load_csv_flow(csv)

    x = np.array([0.05])
    y = np.array([0.0])
    z = np.array([0.0])
    ux, uy, uz = flow(x, y, z)

    assert np.isclose(ux[0], 0.5, atol=1e-3), f"Expected ux≈0.5, got {ux[0]}"
    assert np.isclose(uy[0], 0.0, atol=1e-6)
    assert np.isclose(uz[0], 0.0, atol=1e-6)


def test_load_csv_extrapolation_returns_zero(tmp_path):
    """Points outside the CFD grid convex hull must fall back to 0.0."""
    csv = tmp_path / "field.csv"
    _write_uniform_csv(csv)
    flow = load_csv_flow(csv)

    # Far outside the grid
    x = np.array([999.0])
    y = np.array([999.0])
    z = np.array([999.0])
    ux, uy, uz = flow(x, y, z)

    assert ux[0] == 0.0
    assert uy[0] == 0.0
    assert uz[0] == 0.0


def test_missing_columns_raises(tmp_path):
    """CSV without required columns must raise ValueError with a clear message."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("x,y,z\n0,0,0\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv_flow(bad_csv)


def test_file_not_found_raises():
    """Non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_csv_flow("/does/not/exist.csv")


def test_detect_and_load_csv(tmp_path):
    """detect_and_load must pick the CSV loader for .csv extension."""
    csv = tmp_path / "flow.csv"
    _write_uniform_csv(csv)
    flow = detect_and_load(csv)
    assert isinstance(flow, FlowField)


def test_detect_and_load_unsupported_extension(tmp_path):
    """Unsupported extensions must raise ValueError."""
    f = tmp_path / "flow.xyz"
    f.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        detect_and_load(f)


def test_vtk_raises_import_error_without_pyvista(tmp_path, monkeypatch):
    """load_vtk_flow must raise ImportError when pyvista is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "pyvista", None)  # simulate missing package

    from psat.cfd_loader import load_vtk_flow

    vtk_file = tmp_path / "flow.vtk"
    vtk_file.write_text("dummy")
    with pytest.raises(ImportError, match="pyvista"):
        load_vtk_flow(vtk_file)


def test_flow_field_vectorised(tmp_path):
    """FlowField must accept batched particle arrays without reshaping errors."""
    csv = tmp_path / "field.csv"
    _write_uniform_csv(csv, n=4)
    flow = load_csv_flow(csv)

    x = np.linspace(0.01, 0.09, 20)
    y = np.zeros(20)
    z = np.zeros(20)
    ux, uy, uz = flow(x, y, z)

    assert ux.shape == (20,)
    assert uy.shape == (20,)
    assert uz.shape == (20,)
