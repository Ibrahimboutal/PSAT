import numpy as np
import matplotlib.pyplot as plt

def parabolic_flow(x, y):
    """
    Simple Poiseuille-like flow (like in airways).
    Assumes pipe is centered at y=0 with half-width R=0.01m.
    """
    umax = 0.5
    R = 0.01
    return umax * (1 - (y/R)**2), np.zeros_like(y)

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant, J/K
g = 9.81            # Acceleration due to gravity, m/s^2

class AerosolSimulation:
    def __init__(self, num_particles, dt, total_time, domain_limits,
                 particle_diameter=1e-6, particle_density=1000,
                 fluid_velocity_func=None, T=293.15, mu=1.81e-5):
        """
        Initialize the Monte Carlo simulation for aerosol transport.

        :param num_particles: Number of particles to simulate.
        :param dt: Time step (seconds).
        :param total_time: Total simulation time (seconds).
        :param domain_limits: Tuple of ((xmin, xmax), (ymin, ymax)) defining the boundaries.
        :param particle_diameter: Particle diameter (meters), default 1 um.
        :param particle_density: Particle density (kg/m^3), default 1000 (water-like).
        :param fluid_velocity_func: Function to calculate fluid velocity u_f(x, y). Returns (ux, uy).
        :param T: Temperature (Kelvin), default 293.15 K.
        :param mu: Dynamic viscosity of air (kg/(m*s)), default 1.81e-5.
        """
        self.N = num_particles
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        self.domain_limits = domain_limits
        
        self.T = T
        self.mu = mu
        self.dp = particle_diameter
        self.rho_p = particle_density
        
        # Calculate derived particle properties
        self.mass = particle_density * (np.pi / 6.0) * (self.dp ** 3)
        
        # Cunningham Slip Correction Factor (Cc)
        # Mean free path of air at 293K and 1 atm is approx 66.4 nm
        lambda_air = 6.64e-8 
        knudsen = 2 * lambda_air / self.dp
        self.Cc = 1 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
        
        self.tau = (self.rho_p * self.dp**2 * self.Cc) / (18 * self.mu)  # Relaxation time with slip
        self.D = (k_B * self.T * self.Cc) / (3 * np.pi * self.mu * self.dp)  # Diffusion coefficient with slip
        
        # Fluid velocity field
        if fluid_velocity_func is None:
            self.fluid_velocity_func = lambda x, y: (np.zeros_like(x), np.zeros_like(y))
        else:
            self.fluid_velocity_func = fluid_velocity_func
            
        # Particle state (positions)
        self.positions = np.zeros((self.N, 2))
        self.is_deposited = np.zeros(self.N, dtype=bool)
        self.wall_deposit = np.zeros(self.N, dtype=bool)
        self.bottom_deposit = np.zeros(self.N, dtype=bool)
        
        # Trajectories for visualization
        self.trajectories = np.zeros((self.n_steps + 1, self.N, 2))
        
    def initialize_particles(self, x_coords=None, y_coords=None):
        """Initialize particle positions."""
        ((xmin, xmax), (ymin, ymax)) = self.domain_limits
        
        if x_coords is None:
            # Random uniform placement in the upper half of the domain for testing
            x_coords = np.random.uniform(xmin, xmax, self.N)
        if y_coords is None:
            y_coords = np.random.uniform(ymin + (ymax - ymin)/2, ymax, self.N)
            
        self.positions[:, 0] = x_coords
        self.positions[:, 1] = y_coords
        self.trajectories[0] = self.positions.copy()
        
    def step(self, step_idx):
        """Perform one Euler-Maruyama step."""
        active = ~self.is_deposited
        
        if not np.any(active):
            return  # All particles deposited
        
        x_act = self.positions[active, 0]
        y_act = self.positions[active, 1]
        
        # 1. Advection (Fluid Flow)
        ux, uy = self.fluid_velocity_func(x_act, y_act)
        
        # 2. Settling velocity due to gravity (y-direction)
        v_settling = -self.tau * g
        
        # 3. Brownian motion (Random Walk)
        # Displacement variance is 2*D*dt
        sigma = np.sqrt(2 * self.D * self.dt)
        dW_x = np.random.normal(0, sigma, np.sum(active))
        dW_y = np.random.normal(0, sigma, np.sum(active))
        
        # Update positions
        x_new = x_act + ux * self.dt + dW_x
        y_new = y_act + uy * self.dt + v_settling * self.dt + dW_y
        
        # Check boundary conditions (Deposition)
        ((xmin, xmax), (ymin, ymax)) = self.domain_limits
        
        # For simplicity: particles touching any boundary are considered deposited
        deposited_this_step = (x_new <= xmin) | (x_new >= xmax) | (y_new <= ymin) | (y_new >= ymax)
        
        # Categorize deposition
        hit_bottom = x_new >= xmax
        hit_wall = deposited_this_step & ~hit_bottom
        
        # Clip positions to boundaries so they don't fly off to infinity
        x_new = np.clip(x_new, xmin, xmax)
        y_new = np.clip(y_new, ymin, ymax)
        
        # Write back
        self.positions[active, 0] = x_new
        self.positions[active, 1] = y_new
        
        # Update deposited masks
        active_indices = np.where(active)[0]
        newly_deposited_indices = active_indices[deposited_this_step]
        self.is_deposited[newly_deposited_indices] = True
        
        self.bottom_deposit[active_indices[hit_bottom]] = True
        self.wall_deposit[active_indices[hit_wall]] = True
        
        # Store for history
        self.trajectories[step_idx] = self.positions.copy()
        
    def run(self):
        """Run the full simulation loop."""
        for i in range(1, self.n_steps + 1):
            self.step(i)
        
        # If any particles didn't deposit by the end, fill their remaining trajectory
        for i in range(1, self.n_steps + 1):
            if np.all(self.trajectories[i] == 0):
                # This should not happen since we always copy, but as a safeguard
                pass

    def deposition_efficiency(self):
        """Return total deposition efficiency."""
        return np.sum(self.is_deposited) / self.N
        
    def wall_deposition_fraction(self):
        """Return fraction deposited on walls."""
        return np.sum(self.wall_deposit) / self.N
        
    def bottom_deposition_fraction(self):
        """Return fraction reaching the bottom (deep lung)."""
        return np.sum(self.bottom_deposit) / self.N
        
    def plot_trajectories(self, num_particles=50):
        """Visualize particle trajectories."""
        plt.figure(figsize=(10, 4))
        for i in range(min(num_particles, self.N)):
            plt.plot(self.trajectories[:, i, 0],
                     self.trajectories[:, i, 1], alpha=0.5)
                     
        ((xmin, xmax), (ymin, ymax)) = self.domain_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("Axial Position (x) [m]")
        plt.ylabel("Radial Position (y) [m]")
        plt.title("Particle Trajectories in Airway")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

if __name__ == "__main__":
    # Example simulation block
    print("Running Aerosol Deposition Header Simulations...")
    domain = ((0, 0.1), (-0.01, 0.01)) # length 10cm, width 2cm (radius 1cm)
    
    for dp in [0.5e-6, 1e-6, 5e-6]:
        print(f"\n--- Simulating particle diameter: {dp*1e6:.1f} um ---")
        sim = AerosolSimulation(
            num_particles=200, 
            dt=0.001, 
            total_time=0.4, 
            domain_limits=domain,
            particle_diameter=dp,
            fluid_velocity_func=parabolic_flow
        )
        
        # Inject particles evenly across the inlet
        sim.initialize_particles(
            x_coords=np.zeros(sim.N), 
            y_coords=np.random.uniform(domain[1][0]*0.8, domain[1][1]*0.8, sim.N)
        )
        
        sim.run()
        print(f"Total Deposition Efficiency: {sim.deposition_efficiency():.1%}")
        print(f"Deposited on Walls:          {sim.wall_deposition_fraction():.1%}")
        print(f"Reached Target (Deep Lung):  {sim.bottom_deposition_fraction():.1%}")
        
    # Plot the last simulation trajectories (5um particles)
    print("\nShowing trajectories for 5 um particles...")
    sim.plot_trajectories(num_particles=100)
