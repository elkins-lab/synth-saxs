# synth-saxs

**synth-saxs** is a lightweight Python library for simulating Small-Angle X-ray Scattering (SAXS) profiles from protein coordinates.

Extracted from the [synth-pdb](https://github.com/elkins/synth-pdb) ecosystem, it provides a physically grounded, education-focused engine for reciprocal space simulation.

---

## 🧪 Key Features

*   **Debye Back-Calculation:** O(N²) scattering intensity from atomic coordinates.
*   **Hydration Modeling:** Physically accurate solvent displacement model based on Pavlov & Svergun (1997).
*   **Guinier & Kratky Analysis:** Built-in tools to determine Radius of Gyration ($R_g$) and assess protein compactness.
*   **Educational Clarity:** Explicit, well-commented implementation of form factors and solvent contrast.

## 🚀 Quick Start

### Installation
```bash
pip install synth-saxs[viz]
```

### Basic Usage
```python
import biotite.structure.io.pdb as pdb_io
from synth_saxs import calculate_saxs_profile, plot_saxs_results

# Load a structure
struct = pdb_io.PDBFile.read("protein.pdb").get_structure(model=1)

# Calculate I(q)
q, intensity = calculate_saxs_profile(struct)

# Plot results
plot_saxs_results(q, intensity, plot_type="all")
```

## 📚 Documentation Sections
*   [API Reference](api.md): Detailed function and class signatures.
*   [Scientific Rationale](science.md): The physics and mathematics behind the engine.
*   [Validation Report](validation.md): Comparison against experimental SASBDB data.
*   [Tutorials](tutorials/index.md): Step-by-step guides for common workflows.
