"""
PSAT — Streamlit Web UI
========================
Wraps the aerosol simulation engine in an interactive browser interface.
Deploy in one click to Streamlit Community Cloud:
  https://streamlit.io/cloud
"""
from __future__ import annotations

import io
import time
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from psat.engine import AerosolSimulation, bifurcating_flow_3d
from psat.visualization import plot_deposition, animate_trajectories

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSAT — Aerosol Transport Simulator",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4f8ef7, #a259ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4f8ef7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    div[data-testid="stSidebar"] {
        background: #0f0f1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">💨 PSAT Aerosol Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">3D Monte Carlo · Euler-Maruyama SDE · Numba JIT · Y-bifurcating airway physics</div>',
    unsafe_allow_html=True,
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    st.subheader("Particle Properties")
    num_particles = st.slider("Number of particles", 50, 2000, 300, step=50)
    mean_diameter_um = st.slider("Mean diameter (µm)", 0.1, 10.0, 5.0, step=0.1)
    geo_std_dev = st.slider("Geometric std dev (1.0 = monodisperse)", 1.0, 3.0, 1.5, step=0.1)

    st.subheader("Simulation Setup")
    total_time = st.slider("Simulation time (s)", 0.1, 2.0, 0.4, step=0.05)
    eddy_diff = st.number_input("Eddy diffusivity (m²/s)", value=0.0, format="%.2e", step=1e-6)

    st.subheader("Advanced Forces")
    grad_T_x = st.number_input("∇T  x-component (K/m)", value=0.0, step=10.0)
    E_field_y = st.number_input("E-field y-component (V/m)", value=0.0, step=100.0)
    q_charges = st.number_input("Particle charge (elementary units)", value=0, step=1)

    st.subheader("Outputs")
    generate_animation = st.checkbox("Generate trajectory animation (slower)", value=False)

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
if not run_btn:
    st.info(
        "👈 **Adjust the parameters in the sidebar** and click **Run Simulation** to begin.\n\n"
        "The simulator will compute particle trajectories through a 3D Y-bifurcating airway, "
        "applying Brownian diffusion, gravitational settling, the Cunningham slip factor, and "
        "— if enabled — thermophoresis and electrostatic forces."
    )

    # Show the pre-generated GIF if it exists in the repo
    gif_path = Path("simulation.gif")
    if gif_path.exists():
        st.markdown("### 🎬 Example Simulation")
        st.image(str(gif_path), caption="Pre-computed trajectory animation (300 particles, 0.4 s)")

else:
    mean_diameter_m = mean_diameter_um * 1e-6
    domain_limits = ((0.0, 0.1), (-0.01, 0.01), (-0.01, 0.01))

    with st.spinner(f"Running simulation for {num_particles} particles …"):
        t0 = time.perf_counter()

        sim = AerosolSimulation(
            num_particles=num_particles,
            dt=0.001,
            total_time=total_time,
            domain_limits=domain_limits,
            mean_diameter=mean_diameter_m,
            geo_std_dev=geo_std_dev,
            grad_T=(grad_T_x, 0.0, 0.0),
            E_field=(0.0, E_field_y, 0.0),
            q_charges=int(q_charges),
            eddy_diffusivity=eddy_diff,
            fluid_velocity_func=lambda x, y, z: bifurcating_flow_3d(x, y, z),
            save_trajectories=generate_animation,
        )
        sim.initialize_particles()
        sim.run()

        elapsed = time.perf_counter() - t0

    dep_eff = sim.deposition_efficiency()
    wall_dep = sim.wall_deposition_fraction()
    bot_dep = sim.bottom_deposition_fraction()

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown(f"#### ✅ Simulation complete in **{elapsed:.2f} s** ({num_particles} particles)")
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, value: float, label: str, color: str = "#4f8ef7") -> None:
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{value:.1%}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    metric_card(c1, dep_eff, "Total Deposition")
    metric_card(c2, wall_dep, "Wall Deposition", "#ff6b6b")
    metric_card(c3, bot_dep, "Reached Deep Lung", "#51cf66")
    c4.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color:#ffd43b">{elapsed:.2f}s</div>
            <div class="metric-label">Runtime</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Deposition histogram ──────────────────────────────────────────────────
    st.subheader("📊 Deposition Profile")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ((xmin, xmax), *_) = domain_limits
    wall_positions = sim.positions[sim.wall_deposit]
    if len(wall_positions) > 0:
        ax.hist(
            wall_positions[:, 0],
            bins=40,
            range=(xmin, xmax),
            alpha=0.85,
            color="#ff6b6b",
            edgecolor="#ff8787",
            label=f"Wall Deposits ({int(np.sum(sim.wall_deposit))})",
        )

    ax.set_xlabel("Axial Position (m)", color="#ccc")
    ax.set_ylabel("Particle Count", color="#ccc")
    ax.set_title("Axial Wall Deposition Distribution", color="white")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555")

    st.pyplot(fig)
    plt.close(fig)

    # ── Animation (optional) ─────────────────────────────────────────────────
    if generate_animation and sim.trajectories is not None:
        st.subheader("🎬 Trajectory Animation")
        with st.spinner("Rendering GIF …"):
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                tmp_path = tmp.name
            animate_trajectories(
                sim.trajectories,
                domain_limits,
                num_particles_to_plot=min(150, num_particles),
                save_path=tmp_path,
                fps=20,
            )
            with open(tmp_path, "rb") as f:
                gif_bytes = f.read()

        st.image(gif_bytes, caption="Live particle trajectories through Y-bifurcating airway")
        st.download_button(
            "⬇️ Download animation GIF",
            data=gif_bytes,
            file_name="psat_simulation.gif",
            mime="image/gif",
        )

    # ── Particle size distribution ────────────────────────────────────────────
    if geo_std_dev > 1.0:
        st.subheader("📐 Particle Size Distribution")
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        ax2.hist(
            sim.dp * 1e6,
            bins=40,
            color="#4f8ef7",
            edgecolor="#6fa8ff",
            alpha=0.85,
        )
        ax2.set_xlabel("Diameter (µm)", color="#ccc")
        ax2.set_ylabel("Count", color="#ccc")
        ax2.set_title("Log-Normal Particle Size Distribution", color="white")
        ax2.tick_params(colors="#aaa")
        ax2.spines[:].set_color("#333")
        ax2.grid(True, linestyle="--", alpha=0.3, color="#555")
        st.pyplot(fig2)
        plt.close(fig2)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>"
    "PSAT · Ibrahim Boutal · "
    "<a href='https://github.com/Ibrahimboutal/PSAT' style='color:#4f8ef7;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
