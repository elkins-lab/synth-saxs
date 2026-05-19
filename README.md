# synth-saxs

**synth-saxs** is a lightweight Python library for simulating Small-Angle X-ray Scattering (SAXS) profiles from protein coordinates.

Extracted from the [synth-pdb](https://github.com/elkins/synth-pdb) ecosystem, it provides a physically grounded, education-focused engine for reciprocal space simulation.

## Features
- **Debye Formula**: O(N²) calculation of scattering intensity.
- **Solvent Displacement**: Physically accurate solvent contrast model based on Pavlov & Svergun (1997).
- **Atomic Form Factors**: Standard Waasmaier & Kirfel (1995) coefficients.
- **Visualization**: Built-in support for Kratky and Guinier plots.

## Installation
```bash
pip install synth-saxs
```

## Quick Start
```python
import biotite.structure.io.pdb as pdb_io
from synth_saxs import calculate_saxs_profile

# Load a structure
struct = pdb_io.PDBFile.read("protein.pdb").get_structure(model=1)

# Calculate I(q)
q, I = calculate_saxs_profile(struct)

# Plotting
from synth_saxs import plot_saxs_results
plot_saxs_results(q, I, plot_type="all", output_path="saxs_report.png")
```

## Scientific Rationale
The engine is designed for numerical stability and educational clarity. It correctly handles the delicate balance between atomic contrast and solvent displacement decay to ensure monotonic scattering curves in the Guinier regime.

## References
- Waasmaier, D. & Kirfel, A. (1995). Acta Cryst. A51, 416-431.
- Pavlov, M.Y. & Svergun, D.I. (1997). J. Appl. Cryst. 30, 712-717.
- Svergun, D., et al. (1995). J. Appl. Cryst. 28, 768-773.
