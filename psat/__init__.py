"""
PSAT: 3D Particle Simulation for Aerosol Transport
=================================================

The public API exports the primary simulation engine and CFD data loaders.
"""

from psat.cfd_loader import detect_and_load, load_csv_flow, load_vtk_flow
from psat.engine import AerosolSimulation, bifurcating_flow_3d

__all__ = [
    "AerosolSimulation",
    "bifurcating_flow_3d",
    "detect_and_load",
    "load_csv_flow",
    "load_vtk_flow",
]
