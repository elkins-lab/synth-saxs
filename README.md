# synth-saxs

[![PyPI version](https://img.shields.io/pypi/v/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![Python](https://img.shields.io/pypi/pyversions/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/elkins/synth-saxs/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-saxs/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)

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

## Related Projects

This library is part of the **synth-pdb ecosystem** for synthetic biophysics data generation:

- [synth-pdb](https://github.com/elkins/synth-pdb) — Core protein structure generator (CLI + API)
- [synth-nmr](https://github.com/elkins/synth-nmr) — NMR observables simulator
- [synth-cryo-em](https://github.com/elkins/synth-cryo-em) — Cryo-EM density map simulator
- [synth-dynamics](https://github.com/georgeelkins/synth-dynamics) — ANM/Langevin dynamics engine
- [diff-biophys](https://github.com/elkins/diff-biophys) — Differentiable JAX biophysics kernels
- [diff-ensemble](https://github.com/elkins/diff-ensemble) — IDP structural ensemble generator

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/elkins/synth-saxs). The project uses `ruff` for linting/formatting and `mypy` for type checking — run `pre-commit run --all-files` before submitting.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use synth-saxs in your research, please cite:

```bibtex
@software{synth_saxs,
  author  = {Elkins, George},
  title   = {synth-saxs: SAXS profile simulation from protein coordinates},
  year    = {2024},
  url     = {https://github.com/elkins/synth-saxs},
  version = {0.1.0}
}
```
