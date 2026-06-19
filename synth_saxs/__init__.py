"""
Small-Angle X-ray Scattering (SAXS) Simulation for Structural Biology.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("synth-saxs")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .engine import (
    SaxsSimulator,
    add_noise,
    calculate_p_dist,
    calculate_radius_of_gyration,
    calculate_saxs_profile,
    export_saxs_profile,
    get_form_factor,
    preprocess_structure,
)
from .fitting import (
    calculate_chi_squared,
    fit_profile,
    interpolate_profile,
    load_experimental_data,
)
from .visualization import plot_p_dist, plot_saxs_results

__all__ = [
    "calculate_saxs_profile",
    "calculate_radius_of_gyration",
    "get_form_factor",
    "SaxsSimulator",
    "export_saxs_profile",
    "plot_saxs_results",
    "calculate_p_dist",
    "add_noise",
    "plot_p_dist",
    "preprocess_structure",
    "load_experimental_data",
    "interpolate_profile",
    "fit_profile",
    "calculate_chi_squared",
]
