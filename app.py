"""
PSAT — Streamlit Web UI
========================
Wraps the aerosol simulation engine in an interactive browser interface.
Deploy in one click to Streamlit Community Cloud:
  https://streamlit.io/cloud
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from psat.engine import AerosolSimulation, bifurcating_flow_3d
from psat.visualization import plot_trajectories_plotly
from ui_components import load_css, render_hero_banner, render_metric_card

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSAT — Aerosol Transport Simulator",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
load_css()

# ── Hero banner ───────────────────────────────────────────────────────────────
render_hero_banner()

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
    generate_3d_plot = st.checkbox("Generate interactive 3D trajectories (slower)", value=False)
    run_analytics = st.checkbox("🔬 Tissue Exposure Analytics (clustering)", value=False)

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
            save_trajectories=generate_3d_plot,
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

    render_metric_card(c1, dep_eff, "Total Deposition")
    render_metric_card(c2, wall_dep, "Wall Deposition", "#ff6b6b")
    render_metric_card(c3, bot_dep, "Reached Deep Lung", "#51cf66")
    render_metric_card(c4, elapsed, "Runtime", "#ffd43b", is_time=True)

    # ── Raw Data Export ────────────────────────────────────────────────────────
    df_results = pd.DataFrame(sim.positions, columns=["x (m)", "y (m)", "z (m)"])
    df_results["Deposited"] = sim.is_deposited
    df_results["Wall Deposit"] = sim.wall_deposit
    df_results["Deep Lung Deposit"] = sim.bottom_deposit

    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Raw Particle Data (CSV)",
        data=csv_data,
        file_name="psat_particle_data.csv",
        mime="text/csv",
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

    # ── Interactive 3D Trajectories (optional) ───────────────────────────────
    if generate_3d_plot and sim.trajectories is not None:
        st.subheader("🕸️ Interactive 3D Trajectories")
        with st.spinner("Rendering 3D Plotly graph …"):
            fig_3d = plot_trajectories_plotly(
                sim.trajectories, domain_limits, num_particles_to_plot=min(150, num_particles)
            )
        st.plotly_chart(fig_3d, use_container_width=True)

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

    # ── Tissue Exposure Analytics ─────────────────────────────────────────────
    if run_analytics:
        st.markdown("---")
        st.subheader("🔬 Tissue Exposure Analytics")

        deposited_pos = sim.positions[sim.is_deposited]
        n_dep = len(deposited_pos)

        if n_dep < 2:
            st.warning(
                "Not enough deposited particles for clustering. "
                "Try increasing particle count or simulation time."
            )
        else:
            from psat.analytics import compute_hierarchical_clusters
            from psat.visualization import plot_deposition_clusters_plotly

            with st.spinner("Running hierarchical clustering …"):
                sample_pos, labels = compute_hierarchical_clusters(deposited_pos)
                n_clusters = len(np.unique(labels))

            # Summary metrics
            st.markdown(
                f"**{n_dep} deposited particles** → "
                f"**{n_clusters} hot-spot clusters** identified."
            )
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Deposited Particles", n_dep)
            col_b.metric("Hot-Spot Clusters", n_clusters)
            col_c.metric("Sampled for Clustering", len(sample_pos))

            # 3-D hot-spot scatter
            st.markdown("#### 🗺️ 3D Hot-Spot Map")
            fig_cl = plot_deposition_clusters_plotly(sample_pos, labels, domain_limits)
            st.plotly_chart(fig_cl, use_container_width=True)

            # 2-D axial exposure heat-map
            st.markdown("#### 🌡️ Axial Exposure Density")
            ((xmin, xmax), (ymin, ymax), _) = domain_limits
            fig_hm, ax_hm = plt.subplots(figsize=(10, 3))
            fig_hm.patch.set_facecolor("#0e1117")
            ax_hm.set_facecolor("#0e1117")
            h, xedges, yedges = np.histogram2d(
                sample_pos[:, 0],
                sample_pos[:, 1],
                bins=[60, 30],
                range=[[xmin, xmax], [ymin, ymax]],
            )
            im = ax_hm.imshow(
                h.T,
                origin="lower",
                aspect="auto",
                extent=[xmin, xmax, ymin, ymax],
                cmap="inferno",
                interpolation="bilinear",
            )
            plt.colorbar(im, ax=ax_hm, label="Particle Count", shrink=0.8)
            ax_hm.set_xlabel("Axial Position x (m)", color="#ccc")
            ax_hm.set_ylabel("Radial Position y (m)", color="#ccc")
            ax_hm.set_title("Tissue Exposure Density Map", color="white")
            ax_hm.tick_params(colors="#aaa")
            ax_hm.spines[:].set_color("#333")
            st.pyplot(fig_hm)
            plt.close(fig_hm)

            # Per-cluster stats table
            st.markdown("#### 📋 Per-Cluster Statistics")
            rows = []
            for lbl in np.unique(labels):
                pts = sample_pos[labels == lbl]
                rows.append(
                    {
                        "Cluster": int(lbl),
                        "Particles": len(pts),
                        "Mean x (m)": f"{pts[:, 0].mean():.4f}",
                        "Mean y (m)": f"{pts[:, 1].mean():.4f}",
                        "Mean z (m)": f"{pts[:, 2].mean():.4f}",
                        "x Spread (m)": f"{pts[:, 0].std():.5f}",
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).set_index("Cluster"),
                use_container_width=True,
            )

            # Cluster CSV download
            df_export = pd.DataFrame(sample_pos, columns=["x (m)", "y (m)", "z (m)"])
            df_export["Cluster"] = labels
            st.download_button(
                label="⬇️ Download Cluster Data (CSV)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name="psat_cluster_data.csv",
                mime="text/csv",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>"
    "PSAT · Ibrahim Boutal · "
    "<a href='https://github.com/Ibrahimboutal/PSAT' style='color:#4f8ef7;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
