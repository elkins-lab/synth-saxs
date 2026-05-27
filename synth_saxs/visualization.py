"""
Visualization Module for SAXS Profiles.
Provides plotting capabilities for I(q), Kratky, and Guinier plots.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional Matplotlib Dependency
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_saxs_results(
    q: np.ndarray,
    intensity: np.ndarray,
    title: str = "Synthetic SAXS Profile",
    output_path: str | None = None,
    plot_type: str = "standard",
    rg: float | None = None,
) -> Any:
    """Generate SAXS plots (Standard, Kratky, Guinier, or Porod).

    EDUCATIONAL RATIONALE:
    ----------------------
    SAXS data is a 1D representation of 3D structure. While the raw I(q) curve
    is the fundamental measurement, biological insights are often hidden in
    transformed plots.
    1. Standard (log I vs q): Shows the overall scattering decay.
    2. Kratky (q^2 * I vs q): Highly sensitive to the protein's folding state.
    3. Guinier (ln I vs q^2): Used to measure the overall size (Rg).
    4. Porod (q^4 * I vs q): Used to analyze surface smoothness and compactness.

    Args:
        q: Scattering vector magnitudes.
        intensity: Scattering intensities I(q).
        title: Plot title.
        output_path: If provided, saves plot to file.
        plot_type: 'standard', 'kratky', 'guinier', 'porod', or 'all'.
        rg: Optional Radius of Gyration (A) to overlay on Guinier plot.

    Returns:
        The matplotlib figure object, or None if matplotlib is missing.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not installed. Skipping SAXS visualization.")
        print("\n[INFO]  To enable SAXS visualization, install matplotlib: pip install matplotlib")
        return None

    if plot_type == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _draw_standard_plot(axes[0, 0], q, intensity, title)
        _draw_kratky_plot(axes[0, 1], q, intensity)
        _draw_guinier_plot(axes[1, 0], q, intensity, rg)
        _draw_porod_plot(axes[1, 1], q, intensity)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        if plot_type == "standard":
            _draw_standard_plot(ax, q, intensity, title)
        elif plot_type == "kratky":
            _draw_kratky_plot(ax, q, intensity, title)
        elif plot_type == "guinier":
            _draw_guinier_plot(ax, q, intensity, rg, title)
        elif plot_type == "porod":
            _draw_porod_plot(ax, q, intensity, title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"SAXS plot saved to {output_path}")

    return fig


def plot_p_dist(
    r: np.ndarray,
    p_r: np.ndarray,
    title: str = "Pair Distance Distribution P(r)",
    output_path: str | None = None,
) -> Any:
    """Generate a P(r) plot.

    Args:
        r: Distance values (A).
        p_r: P(r) values.
        title: Plot title.
        output_path: If provided, saves plot to file.

    Returns:
        The matplotlib figure object.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_p_dist_plot(ax, r, p_r, title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"P(r) plot saved to {output_path}")

    return fig


def _draw_standard_plot(ax: Any, q: np.ndarray, intensity: np.ndarray, title: str = "") -> None:
    """Log-linear I(q) vs q plot."""
    ax.semilogy(q, intensity, "b-", linewidth=2, label="I(q)")
    ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    ax.set_ylabel("log I(q)", fontsize=12)
    ax.set_title(title or "SAXS Intensity Profile", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()


def _draw_kratky_plot(ax: Any, q: np.ndarray, intensity: np.ndarray, title: str = "") -> None:
    """Dimensionless-style Kratky plot (q^2 * I(q) vs q)."""
    kratky = (q**2) * intensity
    ax.plot(q, kratky, "r-", linewidth=2, label=r"$q^2 \cdot I(q)$")
    ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    ax.set_ylabel(r"$q^2 \cdot I(q)$", fontsize=12)
    ax.set_title(title or "Kratky Plot (Folding/Flexibility)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def _draw_porod_plot(ax: Any, q: np.ndarray, intensity: np.ndarray, title: str = "") -> None:
    """Porod plot (q^4 * I(q) vs q)."""
    porod = (q**4) * intensity
    ax.plot(q, porod, "m-", linewidth=2, label=r"$q^4 \cdot I(q)$")
    ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    ax.set_ylabel(r"$q^4 \cdot I(q)$", fontsize=12)
    ax.set_title(title or "Porod Plot (Surface)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def _draw_guinier_plot(
    ax: Any, q: np.ndarray, intensity: np.ndarray, rg: float | None = None, title: str = ""
) -> None:
    """Guinier plot (ln(I) vs q^2) for Rg estimation."""
    # Heuristic: try to find a region where q*Rg < 1.3
    # If Rg is not known, use a conservative low-q range
    if rg:
        q_limit = 1.3 / rg
        mask = q <= q_limit
    else:
        # Default to first 10 points or 10%
        mask = np.zeros_like(q, dtype=bool)
        mask[: max(5, len(q) // 10)] = True

    # Filter out q=0 for log
    mask = mask & (q > 1e-6)
    q_low = q[mask]
    i_low = intensity[mask]

    if len(q_low) < 3:
        ax.text(0.5, 0.5, "Insufficient points for Guinier fit", ha="center")
        return

    q2 = q_low**2
    ln_i = np.log(i_low)

    ax.plot(q2, ln_i, "go", markersize=4, label="Low-q Data")

    # Linear fit
    slope, intercept = np.polyfit(q2, ln_i, 1)
    rg_est = np.sqrt(max(0, -3 * slope))
    fit_line = slope * q2 + intercept
    ax.plot(q2, fit_line, "k--", alpha=0.7, label=rf"Fit ($R_g \approx {rg_est:.2f} \AA$)")

    if rg is not None:
        ax.annotate(
            rf"True $R_g = {rg:.2f} \AA$",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            bbox={"boxstyle": "round", "fc": "w", "alpha": 0.5},
        )

    ax.set_xlabel(r"$q^2$ ($\AA^{-2}$)", fontsize=12)
    ax.set_ylabel("ln I(q)", fontsize=12)
    ax.set_title(title or "Guinier Plot", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def _draw_p_dist_plot(ax: Any, r: np.ndarray, p_r: np.ndarray, title: str = "") -> None:
    """P(r) distribution plot."""
    ax.plot(r, p_r, "k-", linewidth=2, label="P(r)")
    ax.fill_between(r, p_r, color="gray", alpha=0.3)
    ax.set_xlabel(r"r ($\AA$)", fontsize=12)
    ax.set_ylabel("P(r)", fontsize=12)
    ax.set_title(title or "Pair Distance Distribution", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
