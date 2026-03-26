import numpy as np
import matplotlib.pyplot as plt
import numba

def bifurcating_flow_3d(x, y, z, L1=0.05, R=0.01, theta=np.pi/6, umax=0.5):
    """
    3D Flow in a Y-Branching pipe.
    Main pipe of length L1 splits into two branches at angle theta.
    """
    ux = np.zeros_like(x)
    uy = np.zeros_like(y)
    uz = np.zeros_like(z)
    
    mask_main = x <= L1
    r2 = y[mask_main]**2 + z[mask_main]**2
    ux[mask_main] = np.where(r2 <= R**2, umax * (1 - r2 / R**2), 0.0)
    
    mask_branch = x > L1
    # Simplified branch flow: constant velocity directed along the branch angle
    ux[mask_branch] = umax * 0.5  # conservation of mass approx
    # Upper branch if y > 0, lower if y < 0
    sign_y = np.sign(y[mask_branch])
    sign_y = np.where(sign_y == 0, 1.0, sign_y) # prevent 0
    uy[mask_branch] = ux[mask_branch] * np.tan(theta) * sign_y
    
    return ux, uy, uz

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant, J/K
g = 9.81            # Acceleration due to gravity, m/s^2

@numba.njit(fastmath=True)
def jitted_physics_core(x_act, y_act, z_act, ux, uy, uz,
                        tau_act, D_act, Z_act, 
                        v_th_x, v_th_y, v_th_z,
                        Ex, Ey, Ez, dt, gravity,
                        xmin, xmax, ymax):
    """
    High-performance compiled subset of the Euler-Maruyama simulation step.
    By extracting this from the class, Numba compiles it directly to C/LLVM.
    """
    n_active = len(x_act)
    
    x_new = np.empty(n_active, dtype=np.float64)
    y_new = np.empty(n_active, dtype=np.float64)
    z_new = np.empty(n_active, dtype=np.float64)
    hit_wall = np.zeros(n_active, dtype=np.bool_)
    hit_bottom = np.zeros(n_active, dtype=np.bool_)
    
    L1 = 0.05
    R_main = ymax
    R_branch = R_main / np.sqrt(2.0)
    theta = np.pi / 6.0
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    
    R_main_sq = R_main**2
    R_branch_sq = R_branch**2
    
    for i in range(n_active):
        # Calculate individual drifts
        v_settling_y = -tau_act[i] * gravity
        total_vx = ux[i] + v_th_x + Z_act[i] * Ex
        total_vy = uy[i] + v_th_y + Z_act[i] * Ey + v_settling_y
        total_vz = uz[i] + v_th_z + Z_act[i] * Ez
        
        # Diffusion
        sigma = np.sqrt(2.0 * D_act[i] * dt)
        dW_x = np.random.normal(0.0, 1.0) * sigma
        dW_y = np.random.normal(0.0, 1.0) * sigma
        dW_z = np.random.normal(0.0, 1.0) * sigma
        
        nx = x_act[i] + total_vx * dt + dW_x
        ny = y_act[i] + total_vy * dt + dW_y
        nz = z_act[i] + total_vz * dt + dW_z
        
        # Boundary Logic
        hw = False
        hb = False
        
        if nx <= L1:
            if ny**2 + nz**2 >= R_main_sq:
                hw = True
        else:
            xb = nx - L1
            yc_up = xb * tan_theta
            yc_down = -xb * tan_theta
            
            dist_up2 = (ny - yc_up)**2 * cos_theta**2 + nz**2
            dist_down2 = (ny - yc_down)**2 * cos_theta**2 + nz**2
            
            if dist_up2 >= R_branch_sq and dist_down2 >= R_branch_sq:
                hw = True
                
        if nx >= xmax:
            hb = True
            
        x_new[i] = nx
        y_new[i] = ny
        z_new[i] = nz
        hit_wall[i] = hw
        hit_bottom[i] = hb
        
    return x_new, y_new, z_new, hit_wall, hit_bottom

class AerosolSimulation:
    def __init__(self, num_particles, dt, total_time, domain_limits,
                 mean_diameter=1e-6, geo_std_dev=1.0, particle_density=1000,
                 fluid_velocity_func=None, T=293.15, mu=1.81e-5,
                 grad_T=(0.0, 0.0, 0.0), E_field=(0.0, 0.0, 0.0), q_charges=0, eddy_diffusivity=0.0):
        """
        Initialize the 3D Monte Carlo simulation for aerosol transport.

        :param domain_limits: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) bounds.
        :param grad_T: Temperature gradient vector (dT/dx, dT/dy, dT/dz) in K/m.
        :param E_field: Electric field vector (Ex, Ey, Ez) in V/m.
        :param q_charges: Number of elementary charges per particle.
        :param eddy_diffusivity: Turbulent eddy diffusivity (m^2/s).
        """
        self.N = num_particles
        self.dt = dt
        self.total_time = total_time
        self.n_steps = int(total_time / dt)
        self.domain_limits = domain_limits
        
        self.T = T
        self.mu = mu
        self.rho_p = particle_density
        
        # Generate Polydisperse Particle Distribution (Log-Normal)
        if geo_std_dev <= 1.0:
            self.dp = np.full(self.N, mean_diameter)
        else:
            mu_ln = np.log(mean_diameter)
            sigma_ln = np.log(geo_std_dev)
            self.dp = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=self.N)
            
        # Particle properties
        self.mass = particle_density * (np.pi / 6.0) * (self.dp ** 3)
        lambda_air = 6.64e-8 
        knudsen = 2 * lambda_air / self.dp
        self.Cc = 1 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
        
        self.tau = (self.rho_p * self.dp**2 * self.Cc) / (18 * self.mu)
        self.D_brownian = (k_B * self.T * self.Cc) / (3 * np.pi * self.mu * self.dp)
        self.D_total = self.D_brownian + eddy_diffusivity # Incorporate turbulent dispersion
        
        # Advanced Forces (External Drifts)
        self.grad_T = np.array(grad_T)
        self.E_field = np.array(E_field)
        
        # Thermophoretic drift (Simplified Brock model approximation)
        # V_th roughly proportional to - grad_T
        K_th_factor = 5e-4 
        self.v_th = - K_th_factor * self.grad_T
        
        # Electrostatic drift
        # Electrical mobility Z = (q * e * Cc) / (3 * pi * mu * dp)
        e_charge = 1.602e-19
        self.Z_mobility = (q_charges * e_charge * self.Cc) / (3 * np.pi * self.mu * self.dp)
        
        # Fluid velocity field
        if fluid_velocity_func is None:
            self.fluid_velocity_func = bifurcating_flow_3d
        else:
            self.fluid_velocity_func = fluid_velocity_func
            
        # State Arrays
        self.positions = np.zeros((self.N, 3))
        self.is_deposited = np.zeros(self.N, dtype=bool)
        self.wall_deposit = np.zeros(self.N, dtype=bool)
        self.bottom_deposit = np.zeros(self.N, dtype=bool)
        self.trajectories = np.zeros((self.n_steps + 1, self.N, 3))
        
    def initialize_particles(self, x_coords=None, y_coords=None, z_coords=None):
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.domain_limits
        if x_coords is None:
            x_coords = np.full(self.N, xmin)
        if y_coords is None or z_coords is None:
            R = min(ymax, zmax)
            r = R * np.sqrt(np.random.uniform(0, 1, self.N))
            theta = np.random.uniform(0, 2*np.pi, self.N)
            y_coords = r * np.cos(theta)
            z_coords = r * np.sin(theta)
            
        self.positions[:, 0] = x_coords
        self.positions[:, 1] = y_coords
        self.positions[:, 2] = z_coords
        self.trajectories[0] = self.positions.copy()
        
    def check_boundaries(self, x, y, z):
        """
        Check if particles hit the Y-Branch complex geometry boundaries.
        Returns boolean masks for (hit_wall, hit_bottom)
        """
        hit_wall = np.zeros_like(x, dtype=bool)
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.domain_limits
        
        L1 = 0.05 # Bifurcation point
        R_main = ymax
        R_branch = R_main / np.sqrt(2) # Conserve area
        theta = np.pi / 6 # 30 degree branch angle
        
        # Section 1: Main Pipe
        mask_main = x <= L1
        hit_wall[mask_main] = (y[mask_main]**2 + z[mask_main]**2 >= R_main**2)
        
        # Section 2: Bifurcation Branches
        mask_branch = x > L1
        x_b = x[mask_branch] - L1
        y_b = y[mask_branch]
        z_b = z[mask_branch]
        
        yc_up = x_b * np.tan(theta)
        yc_down = -x_b * np.tan(theta)
        
        # Distance to center of respective branch cylinder
        dist_up2 = (y_b - yc_up)**2 * np.cos(theta)**2 + z_b**2
        dist_down2 = (y_b - yc_down)**2 * np.cos(theta)**2 + z_b**2
        
        # If the particle is outside BOTH branches, it hit the carved wall
        hit_wall[mask_branch] = (dist_up2 >= R_branch**2) & (dist_down2 >= R_branch**2)
        
        hit_bottom = x >= xmax
        
        return hit_wall, hit_bottom

    def step(self, step_idx):
        active = ~self.is_deposited
        if not np.any(active):
            return
            
        x_act = self.positions[active, 0]
        y_act = self.positions[active, 1]
        z_act = self.positions[active, 2]
        
        # 1. Advection (Python level to support dynamic custom flow functions)
        ux, uy, uz = self.fluid_velocity_func(x_act, y_act, z_act)
        
        # 2 & 3. Advanced Drifts + Diffusion + Boundary Checks (JIT Compiled in C)
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.domain_limits
        
        # Run optimized core
        x_new, y_new, z_new, hit_wall, hit_bottom = jitted_physics_core(
            x_act, y_act, z_act, ux, uy, uz,
            self.tau[active], self.D_total[active], self.Z_mobility[active],
            np.float64(self.v_th[0]), np.float64(self.v_th[1]), np.float64(self.v_th[2]),
            np.float64(self.E_field[0]), np.float64(self.E_field[1]), np.float64(self.E_field[2]),
            self.dt, g,
            xmin, xmax, ymax
        )
        
        deposited_this_step = hit_wall | hit_bottom | (x_new <= xmin)
        
        # Freeze deposited particles at boundary bounds
        x_new = np.clip(x_new, xmin, xmax)
            
        self.positions[active, 0] = x_new
        self.positions[active, 1] = y_new
        self.positions[active, 2] = z_new
        
        active_indices = np.where(active)[0]
        self.is_deposited[active_indices[deposited_this_step]] = True
        self.bottom_deposit[active_indices[hit_bottom & ~hit_wall]] = True
        self.wall_deposit[active_indices[hit_wall]] = True
        
        self.trajectories[step_idx] = self.positions.copy()
        
    def run(self):
        for i in range(1, self.n_steps + 1):
            self.step(i)

    def deposition_efficiency(self):
        return np.sum(self.is_deposited) / self.N
        
    def wall_deposition_fraction(self):
        return np.sum(self.wall_deposit) / self.N
        
    def bottom_deposition_fraction(self):
        return np.sum(self.bottom_deposit) / self.N
