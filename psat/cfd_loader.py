"""
PSAT CFD Loader
===============
Reads external velocity-field data (CSV or VTK) exported from CFD solvers
(OpenFOAM, ANSYS Fluent, etc.) and returns a callable that the
``AerosolSimulation`` engine can use as its ``fluid_velocity_func``.

Supports both steady-state (f(x, y, z)) and time-dependent (f(x, y, z, t))
velocity fields with automatic signature detection and structured grid
optimization (LinearND vs RegularGrid interp).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

_REQUIRED_STEADY_COLS = {"x", "y", "z", "ux", "uy", "uz"}
_REQUIRED_TIME_COLS = {"x", "y", "z", "ux", "uy", "uz", "t"}
_REQUIRED_TIME_COLS_2D = {"x", "y", "u", "v", "t"}


class FlowField:
    """Interpolated 3-D velocity field built from scattered steady-state CFD data.
    Uses Delaunay-based LinearNDInterpolator for unstructured meshes.
    """

    def __init__(
        self,
        points: np.ndarray,
        ux_vals: np.ndarray,
        uy_vals: np.ndarray,
        uz_vals: np.ndarray,
    ) -> None:
        # Detect if the data is 1D, 2D or 3D to avoid Qhull flat-simplex errors
        ranges = points.max(axis=0) - points.min(axis=0)
        self._dim_mask = ranges > 1e-12  # tolerance for floating point epsilon
        self._active_dims = np.where(self._dim_mask)[0]
        pts_filtered = points[:, self._active_dims]

        self._interp_ux = LinearNDInterpolator(pts_filtered, ux_vals, fill_value=0.0)
        self._interp_uy = LinearNDInterpolator(pts_filtered, uy_vals, fill_value=0.0)
        self._interp_uz = LinearNDInterpolator(pts_filtered, uz_vals, fill_value=0.0)

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query the interpolated field at particle positions."""
        raw_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        pts_filtered = raw_pts[:, self._active_dims]

        ux = self._interp_ux(pts_filtered).reshape(x.shape)
        uy = self._interp_uy(pts_filtered).reshape(y.shape)
        uz = self._interp_uz(pts_filtered).reshape(z.shape)
        return ux, uy, uz


class RegularFlowField:
    """Optimized 3-D velocity field for structured grids.
    Uses RegularGridInterpolator (exponentially faster than LinearND).
    """

    def __init__(
        self,
        grid_coords: list[np.ndarray],
        ux_grid: np.ndarray,
        uy_grid: np.ndarray,
        uz_grid: np.ndarray,
        dim_mask: np.ndarray,
    ) -> None:
        self._interp_ux = RegularGridInterpolator(
            grid_coords, ux_grid, bounds_error=False, fill_value=0.0
        )
        self._interp_uy = RegularGridInterpolator(
            grid_coords, uy_grid, bounds_error=False, fill_value=0.0
        )
        self._interp_uz = RegularGridInterpolator(
            grid_coords, uz_grid, bounds_error=False, fill_value=0.0
        )
        self._dim_mask = dim_mask

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Filter particle dimensions to match the grid's intrinsic dimensionality
        raw_pts = [x.ravel(), y.ravel(), z.ravel()]
        query_pts = np.column_stack([raw_pts[i] for i in range(3) if self._dim_mask[i]])

        ux = self._interp_ux(query_pts).reshape(x.shape)
        uy = self._interp_uy(query_pts).reshape(x.shape)
        uz = self._interp_uz(query_pts).reshape(x.shape)
        return ux, uy, uz


class TimeDependentFlowField:
    """Interpolated 4-D velocity field (3D space + 1D time)."""

    def __init__(
        self,
        timesteps: np.ndarray,
        interpolators: dict[float, tuple[Any, ...]],
    ) -> None:
        self.timesteps = timesteps
        self.interpolators = interpolators

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearly interpolate the velocity field in time between CFD snapshots."""
        # Bound check / Clamping
        if t <= self.timesteps[0]:
            return self.interpolators[self.timesteps[0]](x, y, z, t)

        if t >= self.timesteps[-1]:
            return self.interpolators[self.timesteps[-1]](x, y, z, t)

        # Find surrounding timesteps
        idx = np.searchsorted(self.timesteps, t)
        t0 = self.timesteps[idx - 1]
        t1 = self.timesteps[idx]

        alpha = (t - t0) / (t1 - t0)

        f0 = self.interpolators[t0]
        f1 = self.interpolators[t1]

        ux0, uy0, uz0 = f0(x, y, z, t)
        ux1, uy1, uz1 = f1(x, y, z, t)

        # Time-linear interpolation weighted by alpha
        ux = (1 - alpha) * ux0 + alpha * ux1
        uy = (1 - alpha) * uy0 + alpha * uy1
        uz = (1 - alpha) * uz0 + alpha * uz1

        return ux, uy, uz


def wrap_steady_flow(flow_func_3d: Any) -> Any:
    """Wrap a 3-argument function f(x,y,z) into a 4-argument function f(x,y,z,t)."""
    try:
        sig = inspect.signature(flow_func_3d)
        if len(sig.parameters) >= 4:
            return flow_func_3d
    except (ValueError, TypeError):
        pass

    def wrapper(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> Any:
        return flow_func_3d(x, y, z)

    return wrapper


def load_csv_flow(path: str | Path, scale: float = 1.0) -> Any:
    """Load a CFD velocity field from a CSV file (steady or time-dependent)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CFD CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Apply spatial scaling (e.g. mm to m)
    for c in ["x", "y", "z"]:
        if c in df.columns:
            df[c] *= scale

    cols = set(df.columns)
    if "t" in cols:
        return _load_csv_time_dependent(df)

    # Steady-state path
    missing = _REQUIRED_STEADY_COLS - cols
    if missing:
        if {"x", "y", "u", "v"}.issubset(cols):
            df = df.copy()
            df["z"] = 0.0
            df["ux"] = df["u"]
            df["uy"] = df["v"]
            df["uz"] = 0.0
        else:
            raise ValueError(f"CSV is missing required columns: {missing}.")

    return _build_optimal_interpolator(df)


def _build_optimal_interpolator(df: pd.DataFrame) -> Any:
    """Attempts to build a RegularGridInterpolator, defaults to FlowField (LinearND)."""
    x_coords = np.sort(df["x"].unique())
    y_coords = np.sort(df["y"].unique())
    z_coords = np.sort(df["z"].unique())

    n_expected = len(x_coords) * len(y_coords) * len(z_coords)

    # Check if data is structured enough for RegularGrid
    if n_expected == len(df):
        # Determine active dimensions (non-singleton)
        ranges = [np.ptp(c) for c in [x_coords, y_coords, z_coords]]
        mask = [r > 1e-12 for r in ranges]

        # Grid coords for active dimensions
        grid_coords = [c for i, c in enumerate([x_coords, y_coords, z_coords]) if mask[i]]
        grid_shape = [len(c) for c in grid_coords]

        # Sort values to match Meshgrid order (x, y, z)
        df_sorted = df.sort_values(["z", "y", "x"])

        # Pivot values into grids
        ux = df_sorted["ux"].to_numpy().reshape(grid_shape[::-1]).T
        uy = df_sorted["uy"].to_numpy().reshape(grid_shape[::-1]).T
        uz = df_sorted["uz"].to_numpy().reshape(grid_shape[::-1]).T

        return RegularFlowField(grid_coords, ux, uy, uz, np.array(mask))

    # Fallback to Delaunay (Unstructured)
    points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return FlowField(points, df["ux"].to_numpy(), df["uy"].to_numpy(), df["uz"].to_numpy())


def _load_csv_time_dependent(df: pd.DataFrame) -> TimeDependentFlowField:
    """Internal helper to build 4D interpolators from a time-column CSV."""
    cols = set(df.columns)
    if _REQUIRED_TIME_COLS_2D.issubset(cols):
        df = df.copy()
        df["z"] = 0.0
        df["ux"] = df["u"]
        df["uy"] = df["v"]
        df["uz"] = 0.0

    df = df.sort_values("t")
    timesteps = np.sort(df["t"].unique())
    interpolators = {}

    for t in timesteps:
        dft = df[df["t"] == t]
        interpolators[t] = _build_optimal_interpolator(dft)

    return TimeDependentFlowField(timesteps, interpolators)


def load_vtk_flow(path: str | Path, scale: float = 1.0) -> FlowField:
    """Load a CFD velocity field from a VTK or VTU file (currently only steady supported)."""
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            'pyvista is required to load VTK files. Install it with: pip install "psat[cfd]"'
        )

    mesh = pv.read(str(path))
    pts = np.asarray(mesh.points, dtype=np.float64) * scale

    if "U" in mesh.point_data:
        vel = np.asarray(mesh.point_data["U"], dtype=np.float64)
        ux_vals, uy_vals, uz_vals = vel[:, 0], vel[:, 1], vel[:, 2]
    elif {"ux", "uy", "uz"}.issubset(mesh.point_data.keys()):
        ux_vals = np.asarray(mesh.point_data["ux"], dtype=np.float64)
        uy_vals = np.asarray(mesh.point_data["uy"], dtype=np.float64)
        uz_vals = np.asarray(mesh.point_data["uz"], dtype=np.float64)
    else:
        raise KeyError("VTK mesh has no velocity data. Expected array 'U' (N×3) or 'ux, uy, uz'.")

    return FlowField(pts, ux_vals, uy_vals, uz_vals)


def detect_and_load(path: str | Path, scale: float = 1.0) -> Any:
    """Convenience wrapper — picks loader based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv_flow(path, scale=scale)
    if ext in {".vtk", ".vtu"}:
        return load_vtk_flow(path, scale=scale)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv, .vtk, or .vtu.")
