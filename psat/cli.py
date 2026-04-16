from __future__ import annotations

import json

import typer

from psat.engine import AerosolSimulation, bifurcating_flow_3d
from psat.visualization import plot_deposition, plot_trajectories

app = typer.Typer(
    help="PSAT: 3D Particle Simulation for Aerosol Transport",
    add_completion=False,
)



@app.command()
def main(
    num_particles: int = typer.Option(1000, help="Number of particles to simulate"),
    mean_diameter: float = typer.Option(5e-6, help="Geometric Mean particle diameter (meters)"),
    geo_std_dev: float = typer.Option(1.5, help="Geometric Standard Deviation (1.0 = monodisperse)"),
    time: float = typer.Option(0.4, min=0.001, help="Total simulation time in seconds"),
    l1: float = typer.Option(0.05, help="Length of the main pipe (m)"),
    theta: float = typer.Option(0.5235987755982988, help="Bifurcation angle (radians, default pi/6)"),
    grad_t_x: float = typer.Option(0.0, help="Temp gradient x-component (K/m)"),
    grad_t_y: float = typer.Option(0.0, help="Temp gradient y-component (K/m)"),
    grad_t_z: float = typer.Option(0.0, help="Temp gradient z-component (K/m)"),
    e_field_x: float = typer.Option(0.0, help="E-field x-component (V/m)"),
    e_field_y: float = typer.Option(0.0, help="E-field y-component (V/m)"),
    e_field_z: float = typer.Option(0.0, help="E-field z-component (V/m)"),
    q_charges: int = typer.Option(0, help="Number of elementary charges per particle"),
    eddy_diff: float = typer.Option(0.0, help="Turbulent eddy diffusivity (m^2/s)"),
    visualize: str = typer.Option("none", help="Visualization format: none, plot, or animate"),
    output: str = typer.Option("results.json", help="Path to save the simulation statistics"),
    optimize: bool = typer.Option(False, "--optimize", help="Run Optuna Bayesian optimization loop instead of standard simulation"),
    trials: int = typer.Option(50, help="Number of optimization trials (if --optimize is set)")
) -> None:
    """Run the 3D Aerosol Transport Simulation via CLI."""
    if num_particles <= 0:
        typer.secho("Error: Number of particles must be strictly positive.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if mean_diameter <= 0:
        typer.secho("Error: Mean particle diameter must be strictly positive.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if optimize:
        typer.secho(f"🚀 Deploying Agentic Benchmark (Optuna) over {trials} Trials...", fg=typer.colors.MAGENTA, bold=True)
        from psat.optimization import run_optimization

        study = run_optimization(n_trials=trials, num_particles=num_particles)
        best = study.best_params
        val = study.best_value

        typer.secho("\n--- Optimization Complete ---", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"Top Therapy Score Found: {val:.4f}")
        typer.echo(f"Ideal Settings:\n {json.dumps(best, indent=4)}")

        with open("optuna_best_results.json", "w") as f:
            json.dump({"Best Score": val, "Parameters": best}, f, indent=4)
        return

    typer.echo("Initializing PSAT (3D Particle Simulation for Aerosol Transport)...")

    # Domain: Length 10cm (x: 0 to 0.1m), Cylinder radius 1cm (y, z: -0.01 to 0.01)
    domain_limits = ((0.0, 0.1), (-0.01, 0.01), (-0.01, 0.01))

    save_traj = visualize in ["plot", "animate"]

    try:
        sim = AerosolSimulation(
            num_particles=num_particles,
            dt=0.001,
            total_time=time,
            domain_limits=domain_limits,
            mean_diameter=mean_diameter,
            geo_std_dev=geo_std_dev,
            grad_T=(grad_t_x, grad_t_y, grad_t_z),
            E_field=(e_field_x, e_field_y, e_field_z),
            q_charges=q_charges,
            eddy_diffusivity=eddy_diff,
            fluid_velocity_func=lambda x, y, z: bifurcating_flow_3d(x, y, z, L1=l1, theta=theta),
            save_trajectories=save_traj
        )
    except Exception as e:
        typer.secho(f"Initialization Failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("Setting up initial conditions...")
    sim.initialize_particles()

    typer.echo("Running simulation...")
    sim.run(L1=l1, theta=theta)

    dep_eff = sim.deposition_efficiency()
    wall_dep = sim.wall_deposition_fraction()
    bot_dep = sim.bottom_deposition_fraction()

    typer.secho("\n--- Simulation Complete. Statistics ---", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Total Deposition Efficiency: {dep_eff:.1%}")
    typer.echo(f"Deposited on Airways Walls:  {wall_dep:.1%}")
    typer.echo(f"Reached Target (Deep Lung):  {bot_dep:.1%}")
    typer.secho("---------------------------------------\n", fg=typer.colors.GREEN, bold=True)

    # Export results
    results = {
        "num_particles": num_particles,
        "mean_diameter_m": mean_diameter,
        "geo_std_dev": geo_std_dev,
        "total_time_s": time,
        "deposition_efficiency": dep_eff,
        "wall_deposition_fraction": wall_dep,
        "bottom_deposition_fraction": bot_dep
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=4)
    typer.echo(f"Results saved to {output}")

    if visualize != "none":
        typer.echo("Visualizing results...")
        if visualize == "animate":
            from psat.visualization import animate_trajectories
            typer.echo("Rendering animation...")
            animate_trajectories(
                sim.trajectories,
                domain_limits,
                num_particles_to_plot=min(300, num_particles),
                save_path="simulation.gif"
            )
            typer.echo("Animation saved as simulation.gif")
        elif visualize == "plot":
            plot_trajectories(
                sim.trajectories,
                domain_limits,
                num_particles_to_plot=min(300, num_particles),
                save_path="trajectories.png"
            )
            typer.echo("Visualizations saved as trajectories.png and deposition.png")

    if visualize == "none":
        typer.echo("Visualization saved as deposition.png")

    # Always save a deposition plot
    plot_deposition(
        sim.positions,
        domain_limits,
        wall_deposit=sim.wall_deposit,
        bottom_deposit=sim.bottom_deposit,
        save_path="deposition.png"
    )

if __name__ == "__main__":
    app()
