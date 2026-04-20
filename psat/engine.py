"""
PSAT Physics Engine
===================
Implements the 3D Euler-Maruyama stochastic simulation for aerosol transport
through a Y-bifurcating airway geometry.
"""

from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np

# Phase 2 C++ Optimization Extrapolator
# Try resolving Pybind11 Native headers, falling back onto lightning-fast Numba if uncompiled
try:
    import psat_cpp_core

    CPP_ENABLED = True
except ImportError:
    CPP_ENABLED = False

from psat.cfd_loader import wrap_steady_flow
from psat.constants import e_charge, g, k_B, lambda_air

# Type alias for 3-component float tuples used throughout
Vec3 = tuple[float, float, float]


def bifurcating_flow_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    L1: float = 0.05,
    R: float = 0.01,
    theta: float = np.pi / 6,
    umax: float = 0.5,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 3D velocity field in a Y-branching pipe.

    The main pipe of length ``L1`` carries a parabolic (Poiseuille) axial
    flow.  Beyond ``L1`` the pipe splits into two symmetric branches at angle
    ``theta`` — upper (y > 0) and lower (y < 0).

    Parameters
    ----------
    x : np.ndarray
        Axial positions of the active particles (m).
    y : np.ndarray
        Radial-Y positions of the active particles (m).
    z : np.ndarray
        Radial-Z positions of the active particles (m).
    L1 : float, optional
        Axial length of the main pipe before bifurcation (m). Default 0.05.
    R : float, optional
        Radius of the main pipe (m). Default 0.01.
    theta : float, optional
        Half-angle of bifurcation from the x-axis (radians). Default π/6.
    umax : float, optional
        Peak centreline velocity in the main pipe (m/s). Default 0.5.
    t : float, optional
        Current time (s). Default 0.0. Ignored for analytic steady flow.

    Returns
    -------
    ux : np.ndarray
        Axial velocity component (m/s).
    uy : np.ndarray
        Radial-Y velocity component (m/s).
    uz : np.ndarray
        Radial-Z velocity component (m/s).
    """
    ux = np.zeros_like(x)
    uy = np.zeros_like(y)
    uz = np.zeros_like(z)

    # ── Main pipe (Poiseuille profile) ──────────────────────────────────────
    mask_main = x <= L1
    r2 = y[mask_main] ** 2 + z[mask_main] ** 2
    ux[mask_main] = np.where(r2 <= R**2, umax * (1 - r2 / R**2), 0.0)

    # ── Branch region (simplified constant-velocity directional flow) ────────
    mask_branch = x > L1
    ux[mask_branch] = umax * 0.5  # mass-conservation approximation
    sign_y = np.sign(y[mask_branch])
    sign_y = np.where(sign_y == 0, 1.0, sign_y)  # avoid division by zero
    uy[mask_branch] = ux[mask_branch] * np.tan(theta) * sign_y

    return ux, uy, uz


# =========================================================================
# The JIT-compiled Physics loop (Fallback if C++ fails)
# =========================================================================


@numba.njit(fastmath=True)
def jitted_physics_core_numba(
    x_act: np.ndarray,
    y_act: np.ndarray,
    z_act: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    tau_act: np.ndarray,
    D_act: np.ndarray,
    Z_act: np.ndarray,
    v_th_x: float,
    v_th_y: float,
    v_th_z: float,
    Ex: float,
    Ey: float,
    Ez: float,
    dt: float,
    gravity: float,
    xmin: float,
    xmax: float,
    ymax: float,
    L1: float,
    theta: float,
    x_new: np.ndarray,
    y_new: np.ndarray,
    z_new: np.ndarray,
    hit_wall: np.ndarray,
    hit_bottom: np.ndarray,
) -> None:
    n_active = len(x_act)

    R_main = ymax
    R_branch = R_main / np.sqrt(2.0)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    R_main_sq = R_main**2
    R_branch_sq = R_branch**2

    for i in range(n_active):
        # Deterministic drift
        v_settling_y = -tau_act[i] * gravity
        total_vx = ux[i] + v_th_x + Z_act[i] * Ex
        total_vy = uy[i] + v_th_y + Z_act[i] * Ey + v_settling_y
        total_vz = uz[i] + v_th_z + Z_act[i] * Ez

        # Stochastic (Brownian / eddy) diffusion
        sigma = np.sqrt(2.0 * D_act[i] * dt)
        dW_x = np.random.normal(0.0, 1.0) * sigma
        dW_y = np.random.normal(0.0, 1.0) * sigma
        dW_z = np.random.normal(0.0, 1.0) * sigma

        nx = x_act[i] + total_vx * dt + dW_x
        ny = y_act[i] + total_vy * dt + dW_y
        nz = z_act[i] + total_vz * dt + dW_z

        # Boundary detection
        hw = False
        hb = False

        if nx <= L1:
            if ny**2 + nz**2 >= R_main_sq:
                hw = True
        else:
            xb = nx - L1
            yc_up = xb * tan_theta
            yc_down = -xb * tan_theta

            dist_up2 = (ny - yc_up) ** 2 * cos_theta**2 + nz**2
            dist_down2 = (ny - yc_down) ** 2 * cos_theta**2 + nz**2

            if dist_up2 >= R_branch_sq and dist_down2 >= R_branch_sq:
                hw = True

        if nx >= xmax:
            hb = True

        x_new[i] = nx
        y_new[i] = ny
        z_new[i] = nz
        hit_wall[i] = hw
        hit_bottom[i] = hb


# Dynamic Hook
if CPP_ENABLED:
    jitted_physics_core = psat_cpp_core.jitted_physics_core_cpp
else:
    jitted_physics_core = jitted_physics_core_numba


class AerosolSimulation:
    """3D Monte Carlo aerosol transport simulation using Euler-Maruyama integration.

    Simulates polydisperse aerosol particles through a Y-bifurcating airway
    geometry.  The hot physics loop is JIT-compiled via Numba for near-C
    performance.  Advanced forces (thermophoresis, electrostatics, turbulence)
    are supported alongside Stokes drag and Brownian diffusion.

    Parameters
    ----------
    num_particles : int
        Number of Monte Carlo particles (statistical paths) to simulate.
    dt : float
        Time step for Euler-Maruyama integration (s).
    total_time : float
        Total simulation duration (s).
    domain_limits : tuple of tuple of float
        Spatial bounds ``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres.
    mean_diameter : float, optional
        Geometric mean particle diameter (m). Default 1e-6.
    geo_std_dev : float, optional
        Geometric standard deviation of the log-normal size distribution.
        Set to 1.0 for a monodisperse aerosol. Default 1.0.
    particle_density : float, optional
        Particle material density (kg/m³). Default 1000.
    fluid_velocity_func : callable, optional
        Function ``f(x, y, z, t) -> (ux, uy, uz)`` returning np.ndarray
        velocity fields.  Defaults to :func:`bifurcating_flow_3d`.
    T : float, optional
        Fluid temperature (K). Default 293.15.
    mu : float, optional
        Dynamic viscosity of the fluid (Pa·s). Default 1.81e-5 (air at 20 °C).
    grad_T : tuple of float, optional
        Temperature gradient vector ``(dT/dx, dT/dy, dT/dz)`` (K/m).
    E_field : tuple of float, optional
        External electric field vector ``(Ex, Ey, Ez)`` (V/m).
    q_charges : int, optional
        Number of elementary charges per particle.
    eddy_diffusivity : float, optional
        Turbulent eddy diffusivity added to Brownian diffusion (m²/s).
    save_trajectories : bool, optional
        If True, saves full (n_steps × N × 3) position history for animation.
    hygroscopic_growth_rate : float, optional
        Relative rate of diameter growth (1/s). Default 0.0 (no growth).
        e.g., 0.1 means 10% growth per second.
    """

    def __init__(
        self,
        num_particles: int,
        dt: float,
        total_time: float,
        domain_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        mean_diameter: float = 1e-6,
        geo_std_dev: float = 1.0,
        particle_density: float = 1000.0,
        fluid_velocity_func: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        T: float = 293.15,
        mu: float = 1.81e-5,
        grad_T: Vec3 = (0.0, 0.0, 0.0),
        E_field: Vec3 = (0.0, 0.0, 0.0),
        q_charges: int = 0,
        eddy_diffusivity: float = 0.0,
        save_trajectories: bool = False,
        hygroscopic_growth_rate: float = 0.0,
    ) -> None:
        if num_particles <= 0:
            raise ValueError("Number of particles must be strictly positive.")
        if dt <= 0.0:
            raise ValueError("Time step dt must be strictly positive.")
        if total_time <= 0.0:
            raise ValueError("Total simulation time must be strictly positive.")
        if mean_diameter <= 0.0:
            raise ValueError("Mean particle diameter must be strictly positive.")
        if geo_std_dev < 1.0:
            raise ValueError("Geometric standard deviation must be >= 1.0.")

        self.N = num_particles
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        self.domain_limits = domain_limits
        self.growth_rate = hygroscopic_growth_rate
        self.eddy_diff = eddy_diffusivity
        self.q_charges = q_charges

        self.T = T
        self.mu = mu
        self.rho_p = particle_density

        # ── Polydisperse particle size distribution (log-normal) ─────────────
        if geo_std_dev <= 1.0:
            self.dp: np.ndarray = np.full(self.N, mean_diameter)
        else:
            mu_ln = np.log(mean_diameter)
            sigma_ln = np.log(geo_std_dev)
            self.dp = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=self.N)

        # ── Derived particle properties ──────────────────────────────────────
        self.mass: np.ndarray = particle_density * (np.pi / 6.0) * (self.dp**3)
        knudsen: np.ndarray = 2 * lambda_air / self.dp

        # Cunningham slip correction: C_c = 1 + Kn(1.257 + 0.4 exp(-1.1/Kn))
        self.Cc: np.ndarray = 1 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))

        # Stokes relaxation time: τ = ρ_p d_p² C_c / (18 μ)
        self.tau: np.ndarray = (self.rho_p * self.dp**2 * self.Cc) / (18 * self.mu)

        # Brownian diffusivity: D = k_B T C_c / (3 π μ d_p)
        self.D_brownian: np.ndarray = (k_B * self.T * self.Cc) / (3 * np.pi * self.mu * self.dp)
        self.D_total: np.ndarray = self.D_brownian + eddy_diffusivity

        # ── Advanced force parameters ────────────────────────────────────────
        self.grad_T: np.ndarray = np.array(grad_T, dtype=np.float64)
        self.E_field: np.ndarray = np.array(E_field, dtype=np.float64)

        # Simplified Brock thermophoretic drift: V_th ∝ -∇T
        K_th_factor: float = 5e-4
        self.v_th: np.ndarray = -K_th_factor * self.grad_T

        # Electrical mobility: Z = q e C_c / (3 π μ d_p)
        self.Z_mobility: np.ndarray = (q_charges * e_charge * self.Cc) / (
            3 * np.pi * self.mu * self.dp
        )

        # ── Flow field ───────────────────────────────────────────────────────
        if fluid_velocity_func is None:
            self.fluid_velocity_func: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] = (
                wrap_steady_flow(bifurcating_flow_3d)
            )
        else:
            self.fluid_velocity_func = wrap_steady_flow(fluid_velocity_func)

        # ── Pre-allocated Numba output buffers (avoids repeated allocation) ──
        self.x_buf: np.ndarray = np.empty(self.N, dtype=np.float64)
        self.y_buf: np.ndarray = np.empty(self.N, dtype=np.float64)
        self.z_buf: np.ndarray = np.empty(self.N, dtype=np.float64)
        self.hw_buf: np.ndarray = np.empty(self.N, dtype=np.bool_)
        self.hb_buf: np.ndarray = np.empty(self.N, dtype=np.bool_)

        # ── Particle state arrays ────────────────────────────────────────────
        self.positions: np.ndarray = np.zeros((self.N, 3))
        self.is_deposited: np.ndarray = np.zeros(self.N, dtype=bool)
        self.wall_deposit: np.ndarray = np.zeros(self.N, dtype=bool)
        self.bottom_deposit: np.ndarray = np.zeros(self.N, dtype=bool)

        self.save_trajectories: bool = save_trajectories
        if self.save_trajectories:
            self.trajectories: np.ndarray | None = np.zeros((self.n_steps + 1, self.N, 3))
        else:
            self.trajectories = None

    def initialize_particles(
        self,
        x_coords: np.ndarray | None = None,
        y_coords: np.ndarray | None = None,
        z_coords: np.ndarray | None = None,
    ) -> None:
        """Place particles in their initial positions.

        By default particles are injected uniformly at the pipe inlet (x = xmin)
        with radially uniform random positions inside the main pipe cross-section.

        Parameters
        ----------
        x_coords : np.ndarray, optional
            Custom axial starting positions. Shape ``(N,)``.
        y_coords : np.ndarray, optional
            Custom radial-Y starting positions. Shape ``(N,)``.
        z_coords : np.ndarray, optional
            Custom radial-Z starting positions. Shape ``(N,)``.
        """
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.domain_limits
        if x_coords is None:
            x_coords = np.full(self.N, xmin)
        if y_coords is None or z_coords is None:
            R = min(ymax, zmax)
            r = R * np.sqrt(np.random.uniform(0, 1, self.N))
            angle = np.random.uniform(0, 2 * np.pi, self.N)
            y_coords = r * np.cos(angle)
            z_coords = r * np.sin(angle)

        self.positions[:, 0] = x_coords
        self.positions[:, 1] = y_coords
        self.positions[:, 2] = z_coords

        if self.save_trajectories:
            self.trajectories[0] = self.positions.copy()  # type: ignore[index]

    def step(self, step_idx: int, L1: float = 0.05, theta: float = np.pi / 6) -> None:
        """Advance the simulation by one Euler-Maruyama time step.

        Parameters
        ----------
        step_idx : int
            Current step index (used to index the trajectory buffer).
        L1 : float, optional
            Axial bifurcation point (m). Default 0.05.
        theta : float, optional
            Bifurcation half-angle (radians). Default π/6.
        """
        active: np.ndarray = ~self.is_deposited
        if not np.any(active):
            return

        # ── Hygroscopic Growth (Update diameter and dependencies if rate > 0) ──
        if self.growth_rate > 0:
            # d_new = d_old * (1 + rate * dt)
            self.dp[active] *= 1.0 + self.growth_rate * self.dt

            # Recalculate derived physics for active particles
            knudsen = 2 * lambda_air / self.dp[active]
            self.Cc[active] = 1 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
            self.tau[active] = (self.rho_p * self.dp[active] ** 2 * self.Cc[active]) / (
                18 * self.mu
            )

            d_brownian = (k_B * self.T * self.Cc[active]) / (3 * np.pi * self.mu * self.dp[active])
            self.D_total[active] = d_brownian + self.eddy_diff

            self.Z_mobility[active] = (self.q_charges * e_charge * self.Cc[active]) / (
                3 * np.pi * self.mu * self.dp[active]
            )

        x_act = self.positions[active, 0]
        y_act = self.positions[active, 1]
        z_act = self.positions[active, 2]

        # Python-level advection — keeps custom flow functions fully compatible
        t_current = step_idx * self.dt
        ux, uy, uz = self.fluid_velocity_func(x_act, y_act, z_act, t_current)

        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.domain_limits

        # active_indices MUST be computed before n_act uses it
        active_indices: np.ndarray = np.where(active)[0]
        n_act: int = len(active_indices)

        x_new = self.x_buf[:n_act]
        y_new = self.y_buf[:n_act]
        z_new = self.z_buf[:n_act]
        hit_wall = self.hw_buf[:n_act]
        hit_bottom = self.hb_buf[:n_act]

        # JIT-compiled hot path (Numba → LLVM → native machine code)
        jitted_physics_core(
            x_act,
            y_act,
            z_act,
            ux,
            uy,
            uz,
            self.tau[active],
            self.D_total[active],
            self.Z_mobility[active],
            np.float64(self.v_th[0]),
            np.float64(self.v_th[1]),
            np.float64(self.v_th[2]),
            np.float64(self.E_field[0]),
            np.float64(self.E_field[1]),
            np.float64(self.E_field[2]),
            self.dt,
            g,
            xmin,
            xmax,
            ymax,
            L1,
            theta,
            x_new,
            y_new,
            z_new,
            hit_wall,
            hit_bottom,
        )

        deposited_this_step: np.ndarray = hit_wall | hit_bottom | (x_new <= xmin)

        # Clamp positions to prevent particles escaping at outlet
        x_new = np.clip(x_new, xmin, xmax)

        self.positions[active, 0] = x_new
        self.positions[active, 1] = y_new
        self.positions[active, 2] = z_new

        self.is_deposited[active_indices[deposited_this_step]] = True
        self.bottom_deposit[active_indices[hit_bottom & ~hit_wall]] = True
        self.wall_deposit[active_indices[hit_wall]] = True

        if self.save_trajectories:
            self.trajectories[step_idx] = self.positions.copy()  # type: ignore[index]

    def run(self, L1: float = 0.05, theta: float = np.pi / 6) -> None:
        """Run the full simulation from step 1 to ``n_steps``.

        Terminates early if all particles have deposited.

        Parameters
        ----------
        L1 : float, optional
            Axial bifurcation point (m). Default 0.05.
        theta : float, optional
            Bifurcation half-angle (radians). Default π/6.
        """
        for i in range(1, self.n_steps + 1):
            self.step(i, L1=L1, theta=theta)
            if not np.any(~self.is_deposited):
                if self.save_trajectories:
                    self.trajectories = self.trajectories[: i + 1]  # type: ignore[index]
                break

    # ── Deposition statistics ────────────────────────────────────────────────

    def deposition_efficiency(self) -> float:
        """Return the fraction of particles that deposited anywhere.

        Returns
        -------
        float
            Deposition efficiency in [0, 1].
        """
        return float(np.sum(self.is_deposited) / self.N)

    def wall_deposition_fraction(self) -> float:
        """Return the fraction of particles that deposited on airway walls.

        Returns
        -------
        float
            Wall deposition fraction in [0, 1].
        """
        return float(np.sum(self.wall_deposit) / self.N)

    def bottom_deposition_fraction(self) -> float:
        """Return the fraction of particles that reached the outlet (deep lung).

        Returns
        -------
        float
            Outlet (deep-lung) fraction in [0, 1].
        """
        return float(np.sum(self.bottom_deposit) / self.N)
