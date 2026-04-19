"""
PSAT CFD Loader
===============
Reads external velocity-field data (CSV or VTK) exported from CFD solvers
(OpenFOAM, ANSYS Fluent, etc.) and returns a callable that the
``AerosolSimulation`` engine can use as its ``fluid_velocity_func``.

Supports both steady-state (f(x, y, z)) and time-dependent (f(x, y, z, t))
velocity fields with automatic signature detection.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

_REQUIRED_STEADY_COLS = {"x", "y", "z", "ux", "uy", "uz"}
_REQUIRED_TIME_COLS = {"x", "y", "z", "ux", "uy", "uz", "t"}
_REQUIRED_TIME_COLS_2D = {"x", "y", "u", "v", "t"}


class FlowField:
    """Interpolated 3-D velocity field built from scattered steady-state CFD data.

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


class TimeDependentFlowField:
    """Interpolated 4-D velocity field (3D space + 1D time).

    Parameters
    ----------
    timesteps : np.ndarray
        Sorted array of unique time values (s).
    interpolators : dict
        Mapping of ``time -> (ux_interp, uy_interp, uz_interp, dim_mask)``.
    """

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
        raw_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Bound check / Clamping
        if t <= self.timesteps[0]:
            t_ref = self.timesteps[0]
            ux_i, uy_i, uz_i, mask = self.interpolators[t_ref]
            pts = raw_pts[:, mask]
            return (
                ux_i(pts).reshape(x.shape),
                uy_i(pts).reshape(x.shape),
                uz_i(pts).reshape(x.shape),
            )

        if t >= self.timesteps[-1]:
            t_ref = self.timesteps[-1]
            ux_i, uy_i, uz_i, mask = self.interpolators[t_ref]
            pts = raw_pts[:, mask]
            return (
                ux_i(pts).reshape(x.shape),
                uy_i(pts).reshape(x.shape),
                uz_i(pts).reshape(x.shape),
            )

        # Find surrounding timesteps
        idx = np.searchsorted(self.timesteps, t)
        t0 = self.timesteps[idx - 1]
        t1 = self.timesteps[idx]

        alpha = (t - t0) / (t1 - t0)

        ux0, uy0, uz0, m0 = self.interpolators[t0]
        ux1, uy1, uz1, m1 = self.interpolators[t1]

        # Points might have different dimension masks across time (unlikely but safe)
        pts0 = raw_pts[:, m0]
        pts1 = raw_pts[:, m1]

        # Time-linear interpolation weighted by alpha
        ux = (1 - alpha) * ux0(pts0) + alpha * ux1(pts1)
        uy = (1 - alpha) * uy0(pts0) + alpha * uy1(pts1)
        uz = (1 - alpha) * uz0(pts0) + alpha * uz1(pts1)

        return ux.reshape(x.shape), uy.reshape(x.shape), uz.reshape(x.shape)


def wrap_steady_flow(flow_func_3d: Any) -> Any:
    """Wrap a 3-argument function f(x,y,z) into a 4-argument function f(x,y,z,t)."""
    # If it's already a class method or similar with complex signature,
    # checking parameters length is a good heuristic.
    try:
        sig = inspect.signature(flow_func_3d)
        if len(sig.parameters) >= 4:
            return flow_func_3d
    except (ValueError, TypeError):
        # Fail-safe for built-ins or un-inspectable objects
        pass

    def wrapper(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> Any:
        return flow_func_3d(x, y, z)

    return wrapper


def load_csv_flow(path: str | Path) -> Any:
    """Load a CFD velocity field from a CSV file (steady or time-dependent)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CFD CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    cols = set(df.columns)

    if "t" in cols:
        return _load_csv_time_dependent(df)

    # Steady-state path
    missing = _REQUIRED_STEADY_COLS - cols
    if missing:
        # Check if it's 2D steady-state (u,v but no ux,uy,uz)
        if {"x", "y", "u", "v"}.issubset(cols):
            df = df.copy()
            df["z"] = 0.0
            df["ux"] = df["u"]
            df["uy"] = df["v"]
            df["uz"] = 0.0
        else:
            raise ValueError(
                f"CSV is missing required columns: {missing}. Found: {list(df.columns)}"
            )

    points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return FlowField(
        points,
        df["ux"].to_numpy(dtype=np.float64),
        df["uy"].to_numpy(dtype=np.float64),
        df["uz"].to_numpy(dtype=np.float64),
    )


def _load_csv_time_dependent(df: pd.DataFrame) -> TimeDependentFlowField:
    """Internal helper to build 4D interpolators from a time-column CSV."""
    cols = set(df.columns)

    # Normalize 2D data if present
    if _REQUIRED_TIME_COLS_2D.issubset(cols):
        df = df.copy()
        df["z"] = 0.0
        df["ux"] = df["u"]
        df["uy"] = df["v"]
        df["uz"] = 0.0
    elif not _REQUIRED_TIME_COLS.issubset(cols):
        raise ValueError(
            "Time-dependent CSV must have (x,y,u,v,t) or (x,y,z,ux,uy,uz,t). "
            f"Found: {list(df.columns)}"
        )

    df = df.sort_values("t")
    timesteps = np.sort(df["t"].unique())
    interpolators = {}

    for t in timesteps:
        dft = df[df["t"] == t]
        points = dft[["x", "y", "z"]].to_numpy(dtype=np.float64)

        # Detect dimensionality for this timestep
        ranges = points.max(axis=0) - points.min(axis=0)
        mask = np.where(ranges > 1e-12)[0]
        pts_filtered = points[:, mask]

        ux_i = LinearNDInterpolator(
            pts_filtered, dft["ux"].to_numpy(dtype=np.float64), fill_value=0.0
        )
        uy_i = LinearNDInterpolator(
            pts_filtered, dft["uy"].to_numpy(dtype=np.float64), fill_value=0.0
        )
        uz_i = LinearNDInterpolator(
            pts_filtered, dft["uz"].to_numpy(dtype=np.float64), fill_value=0.0
        )
        interpolators[t] = (ux_i, uy_i, uz_i, mask)

    return TimeDependentFlowField(timesteps, interpolators)


def load_vtk_flow(path: str | Path) -> FlowField:
    """Load a CFD velocity field from a VTK or VTU file (currently only steady supported)."""
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


def detect_and_load(path: str | Path) -> Any:
    """Convenience wrapper — picks loader based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv_flow(path)
    if ext in {".vtk", ".vtu"}:
        return load_vtk_flow(path)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv, .vtk, or .vtu.")
