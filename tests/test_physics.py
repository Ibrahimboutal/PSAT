import numpy as np

from psat.constants import g
from psat.engine import AerosolSimulation


def test_terminal_velocity():
    """Test that a particle falls precisely at terminal velocity under gravity."""
    sim = AerosolSimulation(
        num_particles=1,
        dt=0.01,
        total_time=0.1,
        domain_limits=((0, 1), (-1, 1), (-1, 1)),
        mean_diameter=10e-6,
        fluid_velocity_func=lambda x, y, z: (np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)),
        eddy_diffusivity=0.0,
    )
    sim.D_total[:] = 0.0  # Turn off Brownian motion for a deterministic test

    sim.initialize_particles(
        x_coords=np.array([0.0]), y_coords=np.array([0.0]), z_coords=np.array([0.0])
    )

    # Run a single step
    sim.step(1)

    expected_y = -sim.tau[0] * g * sim.dt
    actual_y = sim.positions[0, 1]

    assert np.isclose(actual_y, expected_y, rtol=1e-5), f"Expected y={expected_y}, got {actual_y}"
    assert sim.positions[0, 0] == 0.0, "Expected x to remain 0 with no flow."
    assert sim.positions[0, 2] == 0.0, "Expected z to remain 0 with no flow."


def test_mass_conservation():
    """Test that the number of deposited and active particles always sums to N."""
    sim = AerosolSimulation(
        num_particles=100,
        dt=0.01,
        total_time=0.2,
        domain_limits=((0, 0.05), (-0.01, 0.01), (-0.01, 0.01)),
        fluid_velocity_func=lambda x, y, z: (
            np.ones_like(x) * 0.5,
            np.zeros_like(y),
            np.zeros_like(z),
        ),
    )
    sim.initialize_particles()
    sim.run()

    # Active particles are those not deposited
    active_particles = np.sum(~sim.is_deposited)
    deposited_particles = np.sum(sim.is_deposited)

    assert active_particles + deposited_particles == sim.N, "Mass conservation violated!"


def test_deterministic_bounds_no_diffusion():
    """Test that setting diffusion to 0 produces deterministic bounds outputs."""
    sim = AerosolSimulation(
        num_particles=10,
        dt=0.001,
        total_time=0.01,
        domain_limits=((0, 0.1), (-0.01, 0.01), (-0.01, 0.01)),
        fluid_velocity_func=lambda x, y, z: (
            np.ones_like(x) * 0.1,
            np.zeros_like(y),
            np.zeros_like(z),
        ),
        eddy_diffusivity=0.0,
    )
    sim.D_total[:] = 0.0  # Force zero diffusion

    # Start all particles precisely at the same spot
    x_init = np.zeros(10)
    y_init = np.zeros(10)
    z_init = np.zeros(10)
    sim.initialize_particles(x_coords=x_init, y_coords=y_init, z_coords=z_init)

    for i in range(1, 10):
        sim.step(i)

    # All particles should have exact same coordinates
    assert np.allclose(sim.positions[:, 0], sim.positions[0, 0]), "X coordinates diverged!"
    assert np.allclose(sim.positions[:, 1], sim.positions[0, 1]), "Y coordinates diverged!"
    assert np.allclose(sim.positions[:, 2], sim.positions[0, 2]), "Z coordinates diverged!"
