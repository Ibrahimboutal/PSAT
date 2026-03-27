# PSAT: Particle Simulation for Aerosol Transport

PSAT is a Python-based physics engine designed to simulate and analyze the transport of aerosol particles through pipelines or airways using a Monte Carlo Euler-Maruyama approach.

The simulation accurately models:
- **Fluid Advection**: Uses a parabolic (Poiseuille-like) flow profile.
- **Brownian Diffusion**: Models random walks for sub-micron particles.
- **Gravity Settling**: Accounts for gravitational dropout.
- **Cunningham Slip Correction Factor**: Enhances accuracy for modeling nanoparticle relaxation time and diffusion coefficients. 

## Features

- **Object-Oriented Architecture**: Clean definitions of physics models in `psat/engine.py`.
- **Command-Line Interface (CLI)**: Easily customize parameters without changing code.
- **Statistical Export**: Saves deposition data (wall, target) to JSON files.
- **Advanced Visualizations**: Render both static plots of trajectories/deposition profiles and dynamic animated GIFs of particle transport.

## Installation

Ensure you have Python 3 installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Requires `numpy`, `matplotlib`, and `numba`)*

## Usage

You can run PSAT directly from the terminal. 

```bash
python main.py --help
```

### Command-Line Arguments:
- `--num-particles`: Number of Monte Carlo statistical paths to simulate (default: 1000).
- `--diameter`: Diameter of the aerosol particles in meters. (default: `5e-6` which is 5 um).
- `--time`: Total length of the simulation in seconds (default: 0.4s).
- `--output`: Filepath to output the metrics JSON (default: `results.json`).
- `--animate`: Flag to generate an animated GIF (`simulation.gif`) along with standard execution.

### Example Run
Run a fast simulation of 100 sub-micron (500 nm) particles, generating an animation and a JSON stats output:

```bash
python main.py --num-particles 100 --diameter 0.5e-6 --animate
```

## Outputs

By default, the simulation generates:
1. `trajectories.png` - A static overlay of particle paths through the flow domain.
2. `deposition.png` - A histogram displaying where particles were deposited on the bounds of the domain.
3. `results.json` - A detailed breakdown of efficiency statistics. 
4. `simulation.gif` *(Only if `--animate` is provided)* - A time-series animation mapping particle progression.
