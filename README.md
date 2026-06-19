# synth-saxs

[![codecov](https://codecov.io/gh/elkins-lab/synth-saxs/branch/main/graph/badge.svg)](https://codecov.io/gh/elkins-lab/synth-saxs)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13300/badge)](https://www.bestpractices.dev/projects/13300)
[![PyPI version](https://img.shields.io/pypi/v/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![Python](https://img.shields.io/pypi/pyversions/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20752799-blue)](https://doi.org/10.5281/zenodo.20752799)
[![Tests](https://github.com/elkins-lab/synth-saxs/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/synth-saxs/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)

**synth-saxs** is a lightweight Python library for simulating Small-Angle X-ray Scattering (SAXS) profiles from protein coordinates.

Extracted from the [synth-pdb](https://github.com/elkins/synth-pdb) ecosystem, it provides a physically grounded, education-focused engine for reciprocal space simulation.

---

### 🧪 For Structural Biologists
*   **Guinier & Kratky Analysis:** Built-in plots to determine Radius of Gyration ($R_g$), forward scattering $I(0)$, and assess protein compactness.
*   **Hydration Modeling:** Physically accurate solvent displacement model based on Pavlov & Svergun (1997).

### 🤖 For Machine Learning Researchers
*   **Debye Back-Calculation:** O(N²) scattering intensity from atomic coordinates, suitable as a differentiable-style loss signal for structure validation.
*   **Educational Clarity:** Explicit, well-commented implementation of form factors and solvent contrast — easy to audit and extend.

---

## Features
- **Debye Formula**: O(N²) calculation of scattering intensity.
- **Solvent Displacement**: Physically accurate solvent contrast model based on Pavlov & Svergun (1997).
- **Atomic Form Factors**: Standard Waasmaier & Kirfel (1995) coefficients.
- **Visualization**: Built-in support for Kratky and Guinier plots.

## 📚 Tutorials

Experience **synth-saxs** directly in your browser:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-saxs/blob/main/examples/interactive_tutorials/saxs_profile_generation.ipynb) **SAXS Profile Generation** — Learn how to compute I(q) from coordinates and perform Kratky analysis.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-saxs/blob/main/examples/interactive_tutorials/hydration_shell_analysis.ipynb) **Hydration Shell Analysis** — Interactive visualization of solvent contrast and its effect on perceived protein size.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-saxs/blob/main/examples/interactive_tutorials/end_to_end_validation.ipynb) **End-to-End Validation** — Compare synthetic SAXS data against experimental benchmarks.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-saxs/blob/main/examples/interactive_tutorials/synth_suite_bridge.ipynb) **Synth Suite Bridge** — Demonstrate how to seamlessly link `synth-pdb` structure generation with `synth-saxs`.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-saxs/blob/main/examples/interactive_tutorials/hiv1_rt_hinge_motion.ipynb) **Allosteric Hinge Motions** — Use $P(r)$ distance distributions to detect conformational changes in HIV-1 RT.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-saxs/blob/main/examples/interactive_tutorials/interactive_sandbox.ipynb) **Interactive Sandbox** — Real-time SAXS profile manipulation.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-saxs/blob/main/examples/interactive_tutorials/idp_vs_folded.ipynb) **IDPs vs Folded Proteins** — Contrasting scattering profiles of rigid and disordered proteins.

## Installation
```bash
# Basic installation
pip install synth-saxs

# Installation with visualization support
pip install "synth-saxs[viz]"

# Documentation tooling for contributors
pip install "synth-saxs[docs]"
```

## Command-Line Interface (CLI)

`synth-saxs` provides a CLI for rapid simulation and plotting:

```bash
# Basic simulation
synth-saxs protein.pdb --output profile.dat

# Plotting with Kratky analysis
synth-saxs protein.pdb --plot report.png --plot-type kratky

# Advanced modeling (Hydration Shell + P(r) distribution)
synth-saxs protein.pdb --shell-density 0.03 --p-dist pr.png --p-dist-dat pr.dat
```

### CLI Arguments
- `input`: Path to PDB/mmCIF file.
- `--output`: Save $I(q)$ data to a `.dat` file.
- `--plot`: Path to save a SAXS report image.
- `--shell-density`: Excess hydration shell density (default: 0.0).
- `--p-dist`: Save a plot of the $P(r)$ distribution.
- `--p-dist-dat`: Save raw $P(r)$ data to a `.dat` file.

## Quick Start

### 1. Single Structure Simulation
```python
import biotite.structure.io.pdb as pdb_io
from synth_saxs import calculate_saxs_profile, add_noise

# Load a structure
struct = pdb_io.PDBFile.read("protein.pdb").get_structure(model=1)

# Calculate I(q) and add realistic noise
q, I = calculate_saxs_profile(struct)
I_noisy = add_noise(I, noise_level=0.02)
```

### 2. Ensemble Averaging
The `SaxsSimulator` can handle both Biotite stacks and standard Python lists of structures.
```python
from synth_saxs import SaxsSimulator

# List of different conformation models
models = [model1, model2, model3]

sim = SaxsSimulator(q_max=0.3)
avg_intensity = sim.simulate(models)
```

### 3. Pair Distance Distribution P(r)
```python
from synth_saxs import calculate_p_dist, plot_p_dist

r, pr = calculate_p_dist(struct)
plot_p_dist(r, pr, output_path="p_dist.png")
```

### 4. Visualization
```python
from synth_saxs import plot_saxs_results
plot_saxs_results(q, I, plot_type="all", output_path="saxs_report.png")
```

## Scientific Rationale
The engine is designed for numerical stability and educational clarity. It correctly handles the delicate balance between atomic contrast and solvent displacement decay to ensure monotonic scattering curves in the Guinier regime.

For detailed information on how the engine was validated against peer-reviewed experimental data, see the [Scientific Validation Report](docs/validation.md).

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Community & Security

-   **Code of Conduct**: We expect all participants to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
-   **Security**: Please report security vulnerabilities privately according to our [Security Policy](SECURITY.md).

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use synth-saxs in your research, please cite:

```bibtex
@software{synth_saxs,
  author  = {Elkins, George},
  title   = {synth-saxs: SAXS profile simulation from protein coordinates},
  year    = {2026},
  url     = {https://github.com/elkins/synth-saxs},
  version = {0.1.7}
}
```
