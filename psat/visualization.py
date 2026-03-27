import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_trajectories(trajectories, domain_limits, num_particles_to_plot=100, save_path=None):
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n_particles = trajectories.shape[1]
    plot_idx = np.random.choice(n_particles, min(n_particles, num_particles_to_plot), replace=False)
    
    for idx in plot_idx:
        x = trajectories[:, idx, 0]
        y = trajectories[:, idx, 1]
        z = trajectories[:, idx, 2]
        
        ax.plot(x, y, z, alpha=0.5, linewidth=0.8)
        
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("Axial Position x (m)")
    ax.set_ylabel("Radial Position y (m)")
    ax.set_zlabel("Depth Position z (m)")
    ax.set_title("Aerosol Particle Trajectories in 3D Airway")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_deposition(final_positions, domain_limits, wall_deposit=None, bottom_deposit=None, save_path=None):
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    plt.figure(figsize=(10, 4))
    
    if wall_deposit is not None:
        # Plot where particles hit the walls along the x-axis
        wall_positions = final_positions[wall_deposit]
        plt.hist(wall_positions[:, 0], bins=40, range=(xmin, xmax), 
                 alpha=0.7, color='crimson', edgecolor='black', label="Wall Deposition")
    
    # For legacy behavior support if needed
    if wall_deposit is None:
        floor_deposited = np.isclose(final_positions[:, 1], ymin, atol=1e-3)
        plt.hist(final_positions[floor_deposited, 0], bins=50, range=(xmin, xmax), 
                 alpha=0.7, color='blue', edgecolor='black', label="Floor Deposition")

    plt.xlabel("Axial Position along Airway x (m)")
    plt.ylabel("Number of Deposited Particles")
    plt.title("Wall Deposition Distribution Profile")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def animate_trajectories(trajectories, domain_limits, num_particles_to_plot=100, save_path="simulation.gif", fps=30):
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = domain_limits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n_steps, n_particles, _ = trajectories.shape
    plot_idx = np.random.choice(n_particles, min(n_particles, num_particles_to_plot), replace=False)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("Axial Position x (m)")
    ax.set_ylabel("Radial Position y (m)")
    ax.set_zlabel("Depth Position z (m)")
    ax.set_title("Aerosol Particle Trajectories in 3D Airway")
    
    scat = ax.scatter([], [], [], s=2, alpha=0.5, color='blue')
    
    def update(frame):
        x = trajectories[frame, plot_idx, 0]
        y = trajectories[frame, plot_idx, 1]
        z = trajectories[frame, plot_idx, 2]
        scat._offsets3d = (x, y, z)
        return scat,

    # Reduce frames to save rendering time if simulation is very long
    step = max(1, n_steps // 150) # max ~150 frames
    frames = range(0, n_steps, step)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=1000/fps)
    
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        ani.save(save_path, fps=fps)
        
    plt.close()
