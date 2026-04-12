# Contributing to PSAT

Welcome! We are excited you want to contribute to the Particle Simulation for Aerosol Transport (PSAT) engine.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ibrahimboutal/PSAT.git
   cd PSAT
   ```

2. **Install dependencies:**
   We use `pyproject.toml` to manage dependencies. Install in editable mode along with all development tools (`ruff`, `pytest`, `pre-commit`, etc):
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Pre-Commit Hooks:**
   We enforce formatting using Ruff before any commit is processed. Hook it to your local git:
   ```bash
   pre-commit install
   ```

## Development Workflow

We provide a handy `Makefile` to speed up common tasks:

- **Run the tests:** `make test`
- **Lint the code:** `make lint`
- **Spin up the Web App:** `make run`

Ensure coverage remains high when adding new physics models or parameters!

## Code Architecture

- `psat/engine.py` -> The Stochastic Numba integration loop.
- `psat/cli.py` -> Typer entrypoint for terminal execution.
- `app.py` & `ui_components.py` -> Streamlit Dashboard.
