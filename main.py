import numpy as np
import argparse
import json
from psat.engine import AerosolSimulation
from psat.visualization import plot_trajectories, plot_deposition

def main():
    parser = argparse.ArgumentParser(description="PSAT: 3D Particle Simulation for Aerosol Transport")
    parser.add_argument("--num-particles", type=int, default=1000, help="Number of particles to simulate")
    parser.add_argument("--mean-diameter", type=float, default=5e-6, help="Geometric Mean particle diameter (meters)")
    parser.add_argument("--geo-std-dev", type=float, default=1.5, help="Geometric Standard Deviation (1.0 = monodisperse)")
    parser.add_argument("--time", type=float, default=0.4, help="Total simulation time in seconds")
    
    # Advanced Physics Flags
    parser.add_argument("--grad-t", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Temperature gradient vector (K/m)")
    parser.add_argument("--e-field", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Electric field vector (V/m)")
    parser.add_argument("--q-charges", type=int, default=0, help="Number of elementary charges per particle")
    parser.add_argument("--eddy-diff", type=float, default=0.0, help="Turbulent eddy diffusivity (m^2/s)")
    
    parser.add_argument("--animate", action="store_true", help="Generate an animated trajectory GIF")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save the simulation statistics")
    args = parser.parse_args()

    print("Initializing PSAT (3D Particle Simulation for Aerosol Transport)...")
    
    # Simulation Parameters
    num_particles = args.num_particles
    dt = 0.001  # seconds
    total_time = args.time  # seconds
    
    # Domain: Length 10cm (x: 0 to 0.1m), Cylinder radius 1cm (y, z: -0.01 to 0.01)
    # The new Y-branch bifurcates at x = 0.05m
    domain_limits = ((0.0, 0.1), (-0.01, 0.01), (-0.01, 0.01))
        
    sim = AerosolSimulation(
        num_particles=num_particles,
        dt=dt,
        total_time=total_time,
        domain_limits=domain_limits,
        mean_diameter=args.mean_diameter,
        geo_std_dev=args.geo_std_dev,
        grad_T=tuple(args.grad_t),
        E_field=tuple(args.e_field),
        q_charges=args.q_charges,
        eddy_diffusivity=args.eddy_diff,
        fluid_velocity_func=None  # defaults to the new bifurcating_flow_3d
    )
    
    print("Setting up initial conditions...")
    sim.initialize_particles()
    
    print("Running simulation...")
    sim.run()
    
    dep_eff = sim.deposition_efficiency()
    wall_dep = sim.wall_deposition_fraction()
    bot_dep = sim.bottom_deposition_fraction()

    print("\n--- Simulation Complete. Statistics ---")
    print(f"Total Deposition Efficiency: {dep_eff:.1%}")
    print(f"Deposited on Airways Walls:  {wall_dep:.1%}")
    print(f"Reached Target (Deep Lung):  {bot_dep:.1%}")
    print("---------------------------------------\n")
    
    # Export results
    results = {
        "num_particles": num_particles,
        "mean_diameter_m": args.mean_diameter,
        "geo_std_dev": args.geo_std_dev,
        "total_time_s": total_time,
        "deposition_efficiency": dep_eff,
        "wall_deposition_fraction": wall_dep,
        "bottom_deposition_fraction": bot_dep
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output}")
    
    print("Visualizing results...")
    
    if args.animate:
        from psat.visualization import animate_trajectories
        print("Rendering animation...")
        animate_trajectories(sim.trajectories, domain_limits, num_particles_to_plot=min(300, num_particles), save_path="simulation.gif")
        print("Animation saved as simulation.gif")
    else:
        # Save the trajectory visualization (using 2D projection)
        plot_trajectories(sim.trajectories, domain_limits, num_particles_to_plot=min(300, num_particles), save_path="trajectories.png")
        
        # Save the deposition histogram
        plot_deposition(
            sim.positions, 
            domain_limits, 
            wall_deposit=sim.wall_deposit, 
            bottom_deposit=sim.bottom_deposit, 
            save_path="deposition.png"
        )
        
        print("Visualizations saved as trajectories.png and deposition.png")

if __name__ == "__main__":
    main()
