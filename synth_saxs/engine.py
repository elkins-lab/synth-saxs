"""
# EDUCATIONAL OVERVIEW - SAXS Curve Simulation:
# ---------------------------------------------
# Small-Angle X-ray Scattering (SAXS) is a fundamental technique for studying
# protein structure and dynamics in solution. This module computes synthetic
# scattering curves (I(q) vs q) from atomic coordinates.
#
# SCIENTIFIC PRINCIPLES:
# ----------------------
# 1. The Debye Formula: The scattering intensity I(q) is computed by summing the
#    interference between all pairs of atoms in the molecule.
#    I(q) = sum_i sum_j f_i(q) f_j(q) * sin(q * r_ij) / (q * r_ij)
#    where q is the scattering vector magnitude and r_ij is the distance between
#    atoms i and j.
#
# 2. Atomic Form Factors: Atoms of different elements scatter X-rays with
#    different efficiencies. We use q-dependent form factors approximated by
#    a sum of Gaussians (Waasmaier & Kirfel, 1995).
#
# 3. Solvent Contrast (Solvation Shell): In SAXS, we measure the "excess"
#    scattering of the protein relative to the solvent. We subtract the
#    scattering contribution of the displaced solvent volume (V) for each atom.
#
#    CRITICAL PHYSICAL STABILITY NOTE:
#    The effective scattering factor is f_eff(q) = f_vac(q) - rho_sol * V * exp(-q^2 * R^2 / 10).
#    If the volume V is underestimated (e.g., V=0 for H), the f_eff(q) contrast
#    profile becomes unstable. Specifically, the upward "pressure" from the
#    decaying solvent term can exceed the downward "pressure" from the protein's
#    interferometry, causing non-physical increases in I(q) at low q.
#    Maintaining standard volumes (Pavlov & Svergun, 1997) is essential.
#
# REFERENCES:
# -----------
# - Waasmaier, D. & Kirfel, A. (1995). New analytical scattering-factor
#   functions for free atoms and ions. Acta Cryst. A51, 416-431.
# - Pavlov, M.Y. & Svergun, D.I. (1997). A dataset for testing the
#   algorithms of small-angle scattering data analysis. J. Appl. Cryst. 30, 712-717.
# - Svergun, D., Barberato, C. & Koch, M. H. (1995). CRYSOL - a program to
#   evaluate X-ray solution scattering of biological macromolecules from
#   atomic coordinates. J. Appl. Cryst. 28, 768-773.
"""

import logging
from typing import Any, cast

import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Atomic Form Factor Coefficients (Waasmaier & Kirfel, 1995)
# f(s) = sum_{i=1}^4 a_i * exp(-b_i * s^2) + c, where s = q / (4 * pi)
#
# SCIENTIFIC NOTE - Atomic Volumes:
# ---------------------------------
# Volumes (A^3) are derived from Pavlov & Svergun (1997).
# These "displaced volumes" are critical for the solvent subtraction model.
# Even Hydrogen must have a non-zero volume (~5.15 A^3) to ensure that
# the solvent-corrected form factor f_eff(q) behaves monotonically at low q.
FORM_FACTOR_COEFFS: dict[str, dict[str, Any]] = {
    "H": {
        "a": [0.489918, 0.262477, 0.196767, 0.050479],
        "b": [20.6593, 7.74039, 49.5519, 2.20159],
        "c": 0.00037,
        "volume": 5.15,
    },
    "C": {
        "a": [2.31, 1.02, 1.5886, 0.865],
        "b": [20.8439, 10.2075, 0.5687, 51.6512],
        "c": 0.2156,
        "volume": 16.44,
    },
    "N": {
        "a": [12.2126, 3.1322, 2.0125, 1.1663],
        "b": [0.0057, 9.8933, 28.9974, 0.5826],
        "c": -11.529,
        "volume": 14.0,
    },
    "O": {
        "a": [3.0485, 2.2868, 1.5463, 0.867],
        "b": [13.2771, 5.7011, 0.3239, 32.908],
        "c": 0.2508,
        "volume": 12.0,
    },
    "S": {
        "a": [6.9053, 5.2034, 1.4379, 1.5861],
        "b": [1.4679, 22.2151, 0.2536, 56.172],
        "c": 0.8669,
        "volume": 19.86,
    },
    "P": {
        "a": [6.4345, 4.1791, 1.782, 1.4908],
        "b": [1.9067, 27.157, 0.526, 68.1641],
        "c": 1.1149,
        "volume": 24.4,
    },
    "CL": {
        "a": [11.4604, 3.6801, 1.5923, 0.0567],
        "b": [0.0102, 3.515, 17.151, 62.19],
        "c": -9.7891,
        "volume": 22.45,
    },
    "NA": {
        "a": [4.7626, 3.1736, 1.2674, 1.1128],
        "b": [3.285, 8.8422, 0.3136, 129.424],
        "c": 0.676,
        "volume": 4.0,
    },
    "MG": {
        "a": [5.4204, 2.1735, 1.2269, 2.3073],
        "b": [2.8275, 79.2611, 0.3808, 7.1939],
        "c": 0.8584,
        "volume": 5.0,
    },
    "FE": {
        "a": [11.0531, 7.3817, 4.3975, 1.7543],
        "b": [0.1221, 3.8604, 14.1207, 72.873],
        "c": 1.408,
        "volume": 9.0,
    },
    "ZN": {
        "a": [14.0743, 7.0318, 5.1652, 2.41],
        "b": [0.1164, 2.1569, 10.442, 51.65],
        "c": 1.3152,
        "volume": 10.0,
    },
}


def get_form_factor(element: str, q: np.ndarray) -> np.ndarray:
    """Compute the q-dependent form factor for a given element.

    Args:
        element: Element symbol (e.g. 'C', 'N', 'O').
        q: 1D array of scattering vector magnitudes (Angstroms^-1).

    Returns:
        np.ndarray: Form factor values for each q.
    """
    element = element.upper()
    if element not in FORM_FACTOR_COEFFS:
        # Fallback to Carbon if element unknown
        element = "C"

    coeffs = FORM_FACTOR_COEFFS[element]
    s2 = (q / (4 * np.pi)) ** 2

    f = np.full_like(q, coeffs["c"])
    for a, b in zip(coeffs["a"], coeffs["b"], strict=False):
        f += a * np.exp(-b * s2)

    return f


def calculate_radius_of_gyration(structure: struc.AtomArray) -> float:
    """Calculate the Radius of Gyration (Rg) of a structure.

    Args:
        structure: Biotite AtomArray.

    Returns:
        float: Radius of gyration in Angstroms.
    """
    return float(struc.gyration_radius(structure))


def calculate_saxs_profile(
    structure: struc.AtomArray,
    q_min: float = 0.0,
    q_max: float = 0.5,
    n_points: int = 51,
    include_solvent: bool = True,
    solvent_density: float = 0.334,  # e/A^3 (Water)
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the SAXS profile I(q) for a protein structure.

    This implements the Debye formula with O(N^2) complexity.

    Args:
        structure: Biotite AtomArray (full atom recommended).
        q_min: Minimum q value (default 0.0).
        q_max: Maximum q value (default 0.5).
        n_points: Number of q points.
        include_solvent: If True, subtracts displaced solvent volume.
        solvent_density: Electron density of the solvent.

    Returns:
        Tuple of (q_values, intensity_values).
    """
    n_atoms = structure.array_length()
    logger.info(f"Calculating SAXS profile for {n_atoms} atoms...")

    q = np.linspace(q_min, q_max, n_points)

    # 1. Precompute inter-atomic distances (N x N matrix)
    coords = structure.coord
    if coords.ndim == 3:
        # If passed an AtomArrayStack with 1 model, flatten to 2D
        coords = coords[0]

    # Use scipy for efficient distance calculation
    dist = cdist(coords, coords)

    # 2. Vectorized form factor calculation
    elements = structure.element
    unique_elements = np.unique(elements)
    f_atoms_array = np.zeros((n_atoms, n_points))

    for elem in unique_elements:
        mask = elements == elem
        f_atom = get_form_factor(elem, q)

        if include_solvent:
            # Solvent displacement: f_eff = f_vac - rho_sol * V * exp(-q^2 * R^2 / 10)
            # R is the effective atomic radius: R = (3V / 4pi)^(1/3)
            #
            # SCIENTIFIC NOTE - Monotonicity and Decay:
            # ----------------------------------------
            # The exponent -q^2 * R^2 / K represents the decay of the solvent
            # displacement volume. Using K=6 (Radius of Gyration of a sphere)
            # is physically standard but can lead to non-monotonicity if
            # atomic volumes are small. We use K=10.0 for improved numerical
            # stability across all structure sizes, ensuring the protein's
            # interference always dominates the solvent decay at low q.
            v = FORM_FACTOR_COEFFS.get(elem.upper(), FORM_FACTOR_COEFFS["C"])["volume"]
            decay_rate = ((3 * v) / (4 * np.pi)) ** (2 / 3) / 10.0
            f_sol = solvent_density * v * np.exp(-(q**2) * decay_rate)
            f_atom = f_atom - f_sol

        f_atoms_array[mask] = f_atom

    # 3. Apply Debye formula: I(q) = sum_i f_i^2 + 2 * sum_{i<j} f_i f_j sinc(q r_ij)
    intensity = np.zeros(n_points)

    # Pre-extract upper triangle indices for O(N^2/2) optimization
    # This reduces memory and computation by exploiting symmetry
    triu_i, triu_j = np.triu_indices(n_atoms, k=1)
    r_ij = dist[triu_i, triu_j]

    for i in range(n_points):
        qi = q[i]
        fi = f_atoms_array[:, i]

        if qi < 1e-7:
            # At q=0, sinc(qr) = 1, so I(0) = (sum f_i) ** 2
            intensity[i] = np.sum(fi) ** 2
        else:
            # Self-terms: sum f_i^2
            self_terms = np.sum(fi**2)

            # Cross-terms: 2 * sum_{i<j} f_i * f_j * sinc(q * r_ij)
            # We use vectorization for performance
            f_prod = fi[triu_i] * fi[triu_j]
            sinc_qr = np.sinc((qi * r_ij) / np.pi)
            cross_terms = 2.0 * np.sum(f_prod * sinc_qr)

            intensity[i] = self_terms + cross_terms

    return q, intensity


class SaxsSimulator:
    """Stateful SAXS simulator for ensembles."""

    def __init__(
        self,
        q_min: float = 0.0,
        q_max: float = 0.5,
        n_points: int = 51,
        include_solvent: bool = True,
    ):
        self.q_min = q_min
        self.q_max = q_max
        self.n_points = n_points
        self.include_solvent = include_solvent

    def simulate(self, structure: struc.AtomArray | struc.AtomArrayStack) -> np.ndarray:
        """Computes the averaged SAXS profile for a structure or ensemble."""
        if hasattr(structure, "stack_depth") and structure.stack_depth() > 0:
            # For ensembles, average the intensities
            all_intensities = []
            for i in range(structure.stack_depth()):
                _, intensity = calculate_saxs_profile(
                    structure[i],
                    q_min=self.q_min,
                    q_max=self.q_max,
                    n_points=self.n_points,
                    include_solvent=self.include_solvent,
                )
                all_intensities.append(intensity)

            return cast(np.ndarray, np.mean(all_intensities, axis=0))

        if isinstance(structure, struc.AtomArrayStack) and structure.stack_depth() == 0:
            logger.warning("Attempted to simulate SAXS on an empty ensemble.")
            return np.zeros(self.n_points)

        # Single structure
        _, intensity = calculate_saxs_profile(
            structure,  # type: ignore[arg-type]
            q_min=self.q_min,
            q_max=self.q_max,
            n_points=self.n_points,
            include_solvent=self.include_solvent,
        )
        return intensity


def export_saxs_profile(q: np.ndarray, intensity: np.ndarray, output_file: str) -> None:
    """Export SAXS data to a standard .dat file (q, I, error)."""
    # For synthetic data, we can provide a small dummy error (1% of intensity)
    error = intensity * 0.01
    data = np.column_stack([q, intensity, error])
    header = "Generated by synth-pdb\nq (A^-1)   I(q)       error"
    np.savetxt(output_file, data, header=header, fmt="%.6e")
    logger.info(f"SAXS profile exported to {output_file}")


def calculate_p_dist(
    structure: struc.AtomArray,
    bins: int = 50,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the Pair Distance Distribution Function P(r).

    P(r) represents the distribution of distances between all pairs of atoms
    weighted by their scattering power. It is the Fourier transform of I(q).

    Args:
        structure: Biotite AtomArray.
        bins: Number of bins for the histogram.
        r_max: Maximum distance to consider. If None, uses max inter-atomic distance.

    Returns:
        Tuple of (r_values, p_r_values).
    """
    coords = structure.coord
    if coords.ndim == 3:
        coords = coords[0]

    n_atoms = len(coords)
    dist = cdist(coords, coords)
    triu_i, triu_j = np.triu_indices(n_atoms, k=1)
    r_ij = dist[triu_i, triu_j]

    # Weight by scattering power at q=0: f_i(0) * f_j(0)
    elements = structure.element
    f0 = []
    for elem in elements:
        coeffs = FORM_FACTOR_COEFFS.get(elem.upper(), FORM_FACTOR_COEFFS["C"])
        f0_val = sum(coeffs["a"]) + coeffs["c"]
        f0.append(f0_val)
    f0_arr = np.array(f0)

    weights = f0_arr[triu_i] * f0_arr[triu_j]

    if r_max is None:
        r_max = float(np.max(r_ij) * 1.02)

    hist, bin_edges = np.histogram(r_ij, bins=bins, range=(0, r_max), weights=weights)
    r_vals = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Normalize P(r) such that its integral is 1 (optional, but standard for comparisons)
    # Here we just return the raw weighted histogram
    return r_vals, hist


def add_noise(intensity: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Add proportional Gaussian noise to an intensity profile.

    Args:
        intensity: Input intensity array.
        noise_level: Standard deviation of noise as a fraction of intensity.

    Returns:
        np.ndarray: Noisy intensity values (minimum clamped to 1e-10).
    """
    noise = np.random.normal(0, noise_level * intensity, size=intensity.shape)
    return cast(np.ndarray, np.maximum(1e-10, intensity + noise))
