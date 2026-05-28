# 🧬 synth-saxs: Differentiable SAXS Simulation

[![PyPI version](https://img.shields.io/pypi/v/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/synth-saxs.svg)](https://pypi.org/project/synth-saxs/)
[![Tests](https://github.com/elkins/synth-saxs/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-saxs/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`synth-saxs` is a JAX-powered simulator for back-calculating Small-Angle X-ray Scattering (SAXS) profiles from protein coordinates.

---

### 🧪 For Structural Biologists
*   **Guinier Analysis:** Automatically calculate Radius of Gyration ($R_g$) and forward scattering $I(0)$.
*   **Hydration Modeling:** Explicit support for solvation shell effects on scattering intensity.

### 🤖 For Machine Learning Geeks
*   **Differentiable Debye:** The entire Debye formula is implemented in JAX, allowing gradients to flow from the scattering profile back to atomic positions.
*   **Fast Integration:** JIT-compilable kernels for sub-millisecond SAXS prediction from large structures.

---

## 📦 Installation

```bash
pip install synth-saxs
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
