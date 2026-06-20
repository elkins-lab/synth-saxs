"""
Experimental Data Fitting Module.
Handles loading experimental .dat files, interpolating theoretical curves,
and performing linear least-squares fitting to minimize chi-squared error.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_experimental_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental SAXS data from a 3-column .dat file.

    Format expected: q, I(q), error(q)
    If the file only has 2 columns (q, I(q)), the error is assumed to be 1.0 or
    a proportional constant, though standard SASBDB files have 3 columns.

    Args:
        filepath: Path to the .dat file.

    Returns:
        tuple containing (q, intensity, error) arrays.
    """
    try:
        data = np.loadtxt(filepath, comments=["#", "@"])
    except Exception as e:
        logger.error(f"Failed to load experimental data from {filepath}: {e}")
        raise ValueError(f"Could not parse experimental data file: {e}")

    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError("Experimental data must have at least 2 columns (q, I).")

    q = data[:, 0]
    intensity = data[:, 1]

    if data.shape[1] >= 3:
        error = data[:, 2]
    else:
        logger.warning(
            "No error column found in experimental data. Assuming uniform errors for fitting."
        )
        error = np.ones_like(intensity)

    # Filter out non-positive intensities and errors (which break log plots and chi-square)
    valid_mask = (intensity > 0) & (error > 0)

    return q[valid_mask], intensity[valid_mask], error[valid_mask]


def interpolate_profile(q_exp: np.ndarray, q_calc: np.ndarray, i_calc: np.ndarray) -> np.ndarray:
    """Interpolate the calculated intensity onto the experimental q-grid.

    Args:
        q_exp: Experimental q values.
        q_calc: Calculated q values.
        i_calc: Calculated intensities.

    Returns:
        np.ndarray: Interpolated theoretical intensities matching the q_exp grid.
    """
    # Use logarithmic interpolation for intensities since SAXS varies over many orders of magnitude
    # For robust interpolation, ensure q_calc is strictly increasing (it should be)
    i_calc_log = np.log(np.maximum(i_calc, 1e-12))
    i_interp_log = np.interp(q_exp, q_calc, i_calc_log, left=np.nan, right=np.nan)
    from typing import cast

    return cast(np.ndarray, np.exp(i_interp_log))


def fit_profile(
    i_exp: np.ndarray, err_exp: np.ndarray, i_calc_interp: np.ndarray
) -> tuple[float, float, float]:
    """Fit calculated profile to experimental data via linear least-squares.

    Minimizes chi^2 = sum( ((i_exp - (c * i_calc + k)) / err_exp)^2 )

    Args:
        i_exp: Experimental intensities.
        err_exp: Experimental errors.
        i_calc_interp: Calculated intensities interpolated onto the experimental grid.

    Returns:
        tuple: (c_scale, k_offset, chi_squared)
    """
    # Filter out NaNs (from out-of-bounds interpolation)
    valid = np.isfinite(i_calc_interp) & np.isfinite(i_exp) & np.isfinite(err_exp) & (err_exp > 0)

    if np.sum(valid) < 3:
        raise ValueError("Not enough overlapping q-points to perform a fit.")

    y = i_exp[valid] / err_exp[valid]

    # A matrix for linear least squares: A * x = y
    # Column 0 is the scaled calculated curve: i_calc / err_exp
    # Column 1 is the constant offset: 1.0 / err_exp
    A = np.vstack(
        [i_calc_interp[valid] / err_exp[valid], np.ones(np.sum(valid)) / err_exp[valid]]
    ).T

    # Solve A * x = y
    # lstsq returns: x, residuals, rank, s
    # x is the solution array [c_scale, k_offset]
    solution, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    c_scale = float(solution[0])
    k_offset = float(solution[1])

    i_fit = c_scale * i_calc_interp[valid] + k_offset

    chi_sq = calculate_chi_squared(i_exp[valid], err_exp[valid], i_fit, n_params=2)

    return c_scale, k_offset, chi_sq


def calculate_chi_squared(
    i_exp: np.ndarray, err_exp: np.ndarray, i_fit: np.ndarray, n_params: int = 2
) -> float:
    """Calculate the reduced chi-squared statistic.

    Args:
        i_exp: Experimental intensities.
        err_exp: Experimental errors.
        i_fit: Fitted theoretical intensities.
        n_params: Number of free parameters in the fit (usually 2: scale and offset).

    Returns:
        float: Reduced chi-squared value.
    """
    degrees_of_freedom = len(i_exp) - n_params
    if degrees_of_freedom <= 0:
        return np.inf

    chi_sq = np.sum(((i_exp - i_fit) / err_exp) ** 2)
    reduced_chi_sq = chi_sq / degrees_of_freedom

    return float(reduced_chi_sq)
