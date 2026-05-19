# Scientific Background

## The Debye Formula
The total scattering intensity $I(q)$ is computed by summing the interference between all pairs of atoms $i$ and $j$:

$$I(q) = \sum_i \sum_j f_i(q) f_j(q) \frac{\sin(q r_{ij})}{q r_{ij}}$$

## Solvent Displacement
We use the Pavlov & Svergun (1997) model for solvent subtraction:

$$f_{eff}(q) = f_{vac}(q) - \rho_{sol} V_i \exp\left(-\frac{q^2 V_i^{2/3}}{10}\right)$$
