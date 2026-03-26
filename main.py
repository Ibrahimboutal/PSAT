import numpy as np
import argparse
import json
from psat.engine import AerosolSimulation, parabolic_flow
from psat.visualization import plot_trajectories, plot_deposition

def main():
    parser = argparse.ArgumentParser(description="PSAT: Particle Simulation for Aerosol Transport")
    parser.add_argument("--num-particles", type=int, default=1000, help="Number of particles to simulate")
    parser.add_argument("--diameter", type=float, default=5e-6, help="Particle diameter in meters")
    parser.add_argument("--time", type=float, default=0.4, help="Total simulation time in seconds")
    parser.add_argument("--animate", action="store_true", help="Generate an animated trajectory GIF")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save the simulation statistics")
    args = parser.parse_args()

    print("Initializing PSAT (Particle Simulation for Aerosol Transport)...")
    
    # Simulation Parameters
    num_particles = args.num_particles
    dt = 0.001  # Finer time step for airway scale (seconds)
    total_time = args.time  # seconds
    
    # Domain: Length 10cm (0.1m), width 2cm (radius 1cm) -> coordinates in m
    domain_limits = ((0.0, 0.1), (-0.01, 0.01))
        
    # We create an instance of the simulation engine using the REAL airflow model
    sim = AerosolSimulation(
        num_particles=num_particles,
        dt=dt,
        total_time=total_time,
        domain_limits=domain_limits,
        particle_diameter=args.diameter,
        fluid_velocity_func=parabolic_flow
    )
    
    # Initialize all particles at the inlet (x=0) with random y distribution across the tube
    print("Setting up initial conditions...")
    x_init = np.zeros(num_particles)
    # Inject randomly across the inner 80% of the cross-section
    y_init = np.random.uniform(-0.008, 0.008, num_particles)
    
    sim.initialize_particles(x_init, y_init)
    
    print("Running simulation...")
    sim.run()
    
    # Use the new detailed deposition metrics
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
        "diameter_m": args.diameter,
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
        # Save the trajectory visualization
        plot_trajectories(sim.trajectories, domain_limits, num_particles_to_plot=min(300, num_particles), save_path="trajectories.png")
        
        # Save the deposition histogram using the new tracked arrays
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
