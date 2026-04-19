"""
PSAT CFD Loader
===============
Reads external velocity-field data (CSV or VTK) exported from CFD solvers
(OpenFOAM, ANSYS Fluent, etc.) and returns a callable that the
``AerosolSimulation`` engine can use as its ``fluid_velocity_func``.

Usage
-----
>>> from psat.cfd_loader import load_csv_flow
>>> flow = load_csv_flow("path/to/field.csv")
>>> sim = AerosolSimulation(..., fluid_velocity_func=flow)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_REQUIRED_CSV_COLS = {"x", "y", "z", "ux", "uy", "uz"}


class FlowField:
    """Interpolated 3-D velocity field built from scattered CFD point data.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Sample coordinates ``[x, y, z]`` (m).
    ux_vals, uy_vals, uz_vals : np.ndarray, shape (N,)
        Velocity components at each sample point (m/s).
    """

    def __init__(
        self,
        points: np.ndarray,
        ux_vals: np.ndarray,
        uy_vals: np.ndarray,
        uz_vals: np.ndarray,
    ) -> None:
        self._interp_ux = LinearNDInterpolator(points, ux_vals, fill_value=0.0)
        self._interp_uy = LinearNDInterpolator(points, uy_vals, fill_value=0.0)
        self._interp_uz = LinearNDInterpolator(points, uz_vals, fill_value=0.0)

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query the interpolated field at particle positions.

        Points outside the convex hull of the CFD grid fall back to zero
        velocity (``fill_value=0.0``), matching the analytic flow boundary
        behaviour.

        Parameters
        ----------
        x, y, z : np.ndarray
            Particle position arrays (m).

        Returns
        -------
        ux, uy, uz : np.ndarray
            Interpolated velocity components (m/s).
        """
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        ux = self._interp_ux(pts).reshape(x.shape)
        uy = self._interp_uy(pts).reshape(y.shape)
        uz = self._interp_uz(pts).reshape(z.shape)
        return ux, uy, uz


def load_csv_flow(path: str | Path) -> FlowField:
    """Load a CFD velocity field from a CSV file.

    The CSV must contain at least these columns (header, any order):
    ``x``, ``y``, ``z``, ``ux``, ``uy``, ``uz``

    All other columns are silently ignored.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    FlowField
        Callable interpolated velocity field.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CFD CSV not found: {path}")

    df = pd.read_csv(path)
    missing = _REQUIRED_CSV_COLS - set(df.columns.str.strip().str.lower())
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    df.columns = df.columns.str.strip().str.lower()
    points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return FlowField(
        points,
        df["ux"].to_numpy(dtype=np.float64),
        df["uy"].to_numpy(dtype=np.float64),
        df["uz"].to_numpy(dtype=np.float64),
    )


def load_vtk_flow(path: str | Path) -> FlowField:
    """Load a CFD velocity field from a VTK or VTU file.

    Requires the optional ``pyvista`` package:
    ``pip install "psat[cfd]"``

    The mesh must have a point-data array named ``U`` (shape N×3) or
    separate arrays ``ux``, ``uy``, ``uz``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.vtk`` or ``.vtu`` file.

    Returns
    -------
    FlowField
        Callable interpolated velocity field.

    Raises
    ------
    ImportError
        If ``pyvista`` is not installed.
    KeyError
        If the mesh has no recognised velocity array.
    """
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'pyvista is required to load VTK files. Install it with: pip install "psat[cfd]"'
        ) from exc

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points, dtype=np.float64)

    if "U" in mesh.point_data:
        vel = np.asarray(mesh.point_data["U"], dtype=np.float64)
        ux_vals, uy_vals, uz_vals = vel[:, 0], vel[:, 1], vel[:, 2]
    elif {"ux", "uy", "uz"}.issubset(mesh.point_data.keys()):
        ux_vals = np.asarray(mesh.point_data["ux"], dtype=np.float64)
        uy_vals = np.asarray(mesh.point_data["uy"], dtype=np.float64)
        uz_vals = np.asarray(mesh.point_data["uz"], dtype=np.float64)
    else:
        raise KeyError(
            "VTK mesh has no velocity data. Expected point array 'U' (N×3) "
            "or separate arrays 'ux', 'uy', 'uz'."
        )

    return FlowField(pts, ux_vals, uy_vals, uz_vals)


def detect_and_load(path: str | Path) -> FlowField:
    """Convenience wrapper — picks loader based on file extension.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    FlowField
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv_flow(path)
    if ext in {".vtk", ".vtu"}:
        return load_vtk_flow(path)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv, .vtk, or .vtu.")
