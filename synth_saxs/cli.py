import argparse
import logging
import sys

import biotite.structure.io as strucio
import numpy as np

from synth_saxs.engine import (
    calculate_p_dist,
    calculate_radius_of_gyration,
    calculate_saxs_profile,
    export_saxs_profile,
)
from synth_saxs.visualization import plot_p_dist, plot_saxs_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="synth-saxs: Simulate SAXS profiles from protein coordinates."
    )
    parser.add_argument("input", help="Input structure file (PDB or mmCIF).")
    parser.add_argument("-o", "--output", help="Output .dat file for the SAXS profile.")
    parser.add_argument("--q-max", type=float, default=0.5, help="Maximum q value (A^-1).")
    parser.add_argument("--n-points", type=int, default=51, help="Number of q points.")
    parser.add_argument(
        "--solvent-density", type=float, default=0.334, help="Solvent electron density (e/A^3)."
    )
    parser.add_argument(
        "--shell-density",
        type=float,
        default=0.0,
        help="Excess hydration shell density (e/A^3). Typically 0.02-0.05.",
    )
    parser.add_argument(
        "--no-solvent", action="store_false", dest="include_solvent", help="Disable solvent model."
    )
    parser.add_argument("--plot", help="Output path for the SAXS plot (e.g. report.png).")
    parser.add_argument(
        "--plot-type",
        choices=["standard", "kratky", "guinier", "porod", "all"],
        default="standard",
        help="Type of plot to generate.",
    )
    parser.add_argument("--p-dist", help="Calculate and save P(r) distribution plot (e.g. pr.png).")
    parser.add_argument(
        "--p-dist-dat", help="Calculate and save P(r) distribution data (r, P(r)) to .dat file."
    )

    args = parser.parse_args()

    try:
        # 1. Load Structure
        logger.info(f"Loading structure from {args.input}...")
        structure = strucio.load_structure(args.input)
        if hasattr(structure, "stack_depth") and structure.stack_depth() > 1:
            logger.info(f"Detected stack with {structure.stack_depth()} models. Using model 1.")
            structure = structure[0]

        # 2. Calculate SAXS Profile
        logger.info("Calculating SAXS profile...")
        q, intensity = calculate_saxs_profile(
            structure,
            q_max=args.q_max,
            n_points=args.n_points,
            include_solvent=args.include_solvent,
            solvent_density=args.solvent_density,
            hydration_shell_density=args.shell_density,
        )

        # 3. Export Data
        if args.output:
            export_saxs_profile(q, intensity, args.output)

        # 4. Visualization
        if args.plot:
            rg = calculate_radius_of_gyration(structure)
            plot_saxs_results(
                q, intensity, plot_type=args.plot_type, output_path=args.plot, rg=float(rg)
            )

        # 5. P(r) Calculation
        if args.p_dist or args.p_dist_dat:
            logger.info("Calculating P(r) distribution...")
            r, p_r = calculate_p_dist(structure)

            if args.p_dist_dat:
                data = np.column_stack([r, p_r])
                np.savetxt(args.p_dist_dat, data, header="r (A)      P(r)", fmt="%.6e")
                logger.info(f"P(r) data saved to {args.p_dist_dat}")

            if args.p_dist:
                plot_p_dist(r, p_r, output_path=args.p_dist)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
