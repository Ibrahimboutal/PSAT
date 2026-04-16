from __future__ import annotations

import optuna

from psat.engine import AerosolSimulation, bifurcating_flow_3d


def objective(trial: optuna.Trial, num_particles: int = 500, total_time: float = 0.4) -> float:
    """
    Optuna objective function for evaluating Aerosol therapeutic profiles.

    The score is determined by how precisely the particles deposit into
    the Deep Lung (bottom_fraction) whilst minimizing airway throat
    impaction (wall_fraction).
    """
    # 1. Define hyperparameter search spaces
    mean_diameter_um = trial.suggest_float("mean_diameter_um", 0.1, 10.0)
    geo_std_dev = trial.suggest_float("geo_std_dev", 1.0, 3.0)
    q_charges = trial.suggest_int("q_charges", 0, 10)

    mean_diameter_m = mean_diameter_um * 1e-6
    domain_limits = ((0.0, 0.1), (-0.01, 0.01), (-0.01, 0.01))

    # 2. Build the headless physics engine
    sim = AerosolSimulation(
        num_particles=num_particles,
        dt=0.001,
        total_time=total_time,
        domain_limits=domain_limits,
        mean_diameter=mean_diameter_m,
        geo_std_dev=geo_std_dev,
        grad_T=(0.0, 0.0, 0.0),
        E_field=(0.0, 0.0, 0.0),
        q_charges=q_charges,
        eddy_diffusivity=0.0,
        fluid_velocity_func=lambda x, y, z: bifurcating_flow_3d(x, y, z),
        save_trajectories=False
    )

    # 3. Suppress all but physics
    sim.initialize_particles()
    sim.run()

    # 4. Extract therapeutic profile score
    # We want to MAXIMIZE deep lung (bottom_deposition_fraction)
    # Target Equation: score = bottom_fraction - wall_fraction
    bottom_dep = sim.bottom_deposition_fraction()
    wall_dep = sim.wall_deposition_fraction()

    score = bottom_dep - wall_dep
    return score

def run_optimization(n_trials: int = 50, num_particles: int = 500) -> optuna.Study:
    """
    Spins up the Bayesian optimization loop over N simulated trials.
    """
    study = optuna.create_study(direction="maximize", study_name="Aerosol_Optimization")

    # Wrap the objective parameters into a strict Optuna callable
    def func(trial: optuna.Trial) -> float:
        return objective(trial, num_particles=num_particles)

    study.optimize(func, n_trials=n_trials)
    return study
