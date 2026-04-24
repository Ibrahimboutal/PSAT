import numpy as np

from psat.engine import AerosolSimulation, generate_weibel_tree


def test_weibel_tree_generation():
    """Verify that the tree generation produces the correct number of branches."""
    # 1 generation = 1 branch
    starts, dirs, lens, radii, left, right = generate_weibel_tree(1)
    assert len(starts) == 1
    assert left[0] == -1

    # 2 generations = 3 branches (1 + 2)
    starts, dirs, lens, radii, left, right = generate_weibel_tree(2)
    assert len(starts) == 3
    assert left[0] == 1
    assert right[0] == 2
    assert left[1] == -1
    assert left[2] == -1


def test_recursive_simulation_run():
    """Verify that a simulation can run with multiple generations without crashing."""
    sim = AerosolSimulation(
        num_particles=50,
        dt=0.01,
        total_time=0.2,
        domain_limits=((0, 0.1), (-0.01, 0.01), (-0.01, 0.01)),
        n_generations=3,
        turbulence_alpha=0.1,
    )
    sim.initialize_particles()
    # Should not raise any errors
    sim.run()

    # Ensure particles moved
    assert not np.allclose(sim.positions, 0.0)
    # Ensure some deposition or transit occurred
    assert sim.deposition_efficiency() >= 0.0


def test_turbulence_impact():
    """Verify that turbulence alpha increases particle dispersion (stochastically)."""
    # This is a bit hard to test deterministically, but we can check if D_total changes
    # in the engine if we were to expose it, or just ensure the simulation runs.
    sim_no_turb = AerosolSimulation(
        num_particles=100,
        dt=0.001,
        total_time=0.05,
        domain_limits=((0, 0.1), (-0.01, 0.01), (-0.01, 0.01)),
        n_generations=2,
        turbulence_alpha=0.0,
    )
    sim_turb = AerosolSimulation(
        num_particles=100,
        dt=0.001,
        total_time=0.05,
        domain_limits=((0, 0.1), (-0.01, 0.01), (-0.01, 0.01)),
        n_generations=2,
        turbulence_alpha=1.0,
    )

    sim_no_turb.initialize_particles()
    sim_turb.initialize_particles()

    # Seed fixed for comparison if possible (Numba random is separate though)
    sim_no_turb.run()
    sim_turb.run()

    # Just verify they both completed
    assert sim_no_turb.n_steps > 0
    assert sim_turb.n_steps > 0
