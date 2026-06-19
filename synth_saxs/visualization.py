"""
Visualization Module for SAXS Profiles.
Provides plotting capabilities for I(q), Kratky, and Guinier plots.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
PLOT_TYPES = {"standard", "kratky", "guinier", "porod", "all"}

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
    exp_data: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None,
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
        exp_data: Optional tuple from fitting (q_exp, i_exp, err_exp, c_scale, k_offset).

    Returns:
        The matplotlib figure object, or None if matplotlib is missing.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not installed. Skipping SAXS visualization.")
        print("\n[INFO]  To enable SAXS visualization, install matplotlib: pip install matplotlib")
        return None
    if plot_type not in PLOT_TYPES:
        accepted = ", ".join(sorted(PLOT_TYPES))
        raise ValueError(f"plot_type must be one of: {accepted}.")

    if plot_type == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _draw_standard_plot(axes[0, 0], q, intensity, title, exp_data)
        _draw_kratky_plot(axes[0, 1], q, intensity)
        _draw_guinier_plot(axes[1, 0], q, intensity, rg)
        _draw_porod_plot(axes[1, 1], q, intensity)
    else:
        if plot_type == "standard" and exp_data is not None:
            # Create a 2-row layout: main plot (top, larger), residuals (bottom, smaller)
            fig = plt.figure(figsize=(8, 7))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(gs[0])
            ax_res = fig.add_subplot(gs[1], sharex=ax_main)
            _draw_standard_plot(ax_main, q, intensity, title, exp_data, ax_res)
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


def _draw_standard_plot(
    ax: Any,
    q: np.ndarray,
    intensity: np.ndarray,
    title: str = "",
    exp_data: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None,
    ax_res: Any = None,
) -> None:
    """Log-linear I(q) vs q plot, optionally with experimental data overlay and residuals."""
    if exp_data is not None:
        q_exp, i_exp, err_exp, c, k = exp_data

        # Plot experimental data
        ax.errorbar(
            q_exp,
            i_exp,
            yerr=err_exp,
            fmt="ko",
            markersize=3,
            alpha=0.5,
            label="Experimental",
            zorder=1,
        )

        # Plot fitted theoretical curve
        from synth_saxs.fitting import interpolate_profile

        i_calc_interp = interpolate_profile(q_exp, q, intensity)
        i_fit = c * i_calc_interp + k

        # We also plot the continuous theoretical curve shifted by c and k
        intensity_scaled = c * intensity + k
        ax.semilogy(q, intensity_scaled, "r-", linewidth=2, label="Theoretical Fit", zorder=2)

        if ax_res is not None:
            # Calculate normalized residuals
            residuals = (i_exp - i_fit) / err_exp
            ax_res.plot(q_exp, residuals, "k.", markersize=3, alpha=0.5)
            ax_res.axhline(0, color="r", linestyle="--", linewidth=1)
            ax_res.set_ylabel(r"$\Delta I / \sigma$", fontsize=12)
            ax_res.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
            ax_res.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(labelbottom=False)  # hide x labels on main plot
        else:
            ax.set_xlabel(r"q ($\AA^{-1}$)", fontsize=12)
    else:
        ax.semilogy(q, intensity, "b-", linewidth=2, label="Theoretical I(q)")
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
