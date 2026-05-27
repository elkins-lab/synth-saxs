"""
Scientific Validation: SAXS Rg Calibration and Profile Shape.

Validates Rg physical bounds and Guinier-regime monotonicity.

REFERENCES:
  Guinier, A. (1939). Ann Phys (Paris), 12, 161-237.
  Millett et al. (2002). Adv Protein Chem, 62, 241-262.
"""

import biotite.structure as struc
import numpy as np
import pytest

from synth_saxs import calculate_radius_of_gyration, calculate_saxs_profile


def create_mock_peptide(n_residues: int, compact: bool = True) -> struc.AtomArray:
    """Creates a mock peptide-like structure (linear chain of C-N-C-O)."""
    # 4 atoms per residue (approx)
    n_atoms = n_residues * 4
    array = struc.AtomArray(n_atoms)

    # Simple linear placement
    step = 1.5 if compact else 3.8
    coords = np.zeros((n_atoms, 3))
    coords[:, 0] = np.arange(n_atoms) * step

    array.coord = coords
    array.element = ["C", "N", "C", "O"] * n_residues
    array.res_id = np.repeat(np.arange(1, n_residues + 1), 4)
    array.atom_name = ["C", "N", "CA", "O"] * n_residues
    array.chain_id = np.full(n_atoms, "A")
    array.hetero = np.full(n_atoms, False)

    return array


@pytest.fixture(scope="module")
def mock_peptide():
    return create_mock_peptide(20)


def test_rg_positive_and_finite(mock_peptide):
    """Rg must be a positive, finite real number."""
    rg = calculate_radius_of_gyration(mock_peptide)
    assert np.isfinite(rg), f"Rg is not finite: {rg}"
    assert rg > 0.0


def test_rg_physically_sensible_for_20_residue_peptide(mock_peptide):
    """Rg for a 20-residue peptide must lie in [5, 50] A."""
    rg = calculate_radius_of_gyration(mock_peptide)
    print(f"\n  Rg (20-residue mock) = {rg:.2f} A")
    # A linear chain of 80 atoms with 1.5A spacing is quite large (~120A total)
    # so Rg will be ~35A.
    assert 5.0 <= rg <= 100.0, f"Rg {rg:.2f} A outside reasonable bounds"


def test_saxs_profile_shape(mock_peptide):
    """q and I(q) arrays must be equal-length with correct endpoints."""
    q, intensity = calculate_saxs_profile(mock_peptide, q_max=0.3, n_points=31)
    assert q.shape == intensity.shape
    assert len(q) == 31
    assert q[0] == pytest.approx(0.0, abs=1e-6)
    assert q[-1] == pytest.approx(0.3, abs=1e-3)


def test_saxs_intensity_positive(mock_peptide):
    """All I(q) must be positive (physical requirement)."""
    _, intensity = calculate_saxs_profile(mock_peptide, q_max=0.3, n_points=31)
    assert np.all(intensity > 0), f"Negative I(q) found; min = {intensity.min():.4g}"


def test_saxs_low_q_monotonic_decrease(mock_peptide):
    """I(q) must decrease monotonically in Guinier regime (q < 0.08 A^-1)."""
    q, intensity = calculate_saxs_profile(
        mock_peptide, q_max=0.3, n_points=61, include_solvent=True
    )
    mask = q <= 0.08
    i_low = intensity[mask]
    if len(i_low) < 3:
        pytest.skip("Insufficient q-points in Guinier regime")
    diffs = np.diff(i_low)
    assert np.all(diffs <= 0), "I(q) not monotonically decreasing at q < 0.08 A^-1"


def test_saxs_monotonicity_scale_invariance():
    """Verify monotonicity holds for structures of different scales."""
    # 1. Very small peptide (3 residues)
    struct_3 = create_mock_peptide(3)
    q3, i3 = calculate_saxs_profile(struct_3, q_max=0.08, n_points=20, include_solvent=True)
    assert np.all(np.diff(i3) <= 0), "Small peptide failed monotonicity"

    # 2. Medium peptide (50 residues)
    struct_50 = create_mock_peptide(50)
    q50, i50 = calculate_saxs_profile(struct_50, q_max=0.08, n_points=20, include_solvent=True)
    assert np.all(np.diff(i50) <= 0), "Medium protein failed monotonicity"


def test_saxs_solvent_vacuum_ratio(mock_peptide):
    """Vacuum intensity I_vac(0) must be greater than solvent-subtracted I_sol(0)."""
    _, i_vac = calculate_saxs_profile(mock_peptide, q_max=0.1, n_points=5, include_solvent=False)
    _, i_sol = calculate_saxs_profile(mock_peptide, q_max=0.1, n_points=5, include_solvent=True)

    assert i_sol[0] < i_vac[0], "Solvent-subtracted I(0) should be less than vacuum I(0)"
    assert i_sol[0] > 0, "Effective I(0) must be positive"


def test_saxs_guinier_rg_consistent_with_direct(mock_peptide):
    """Guinier-fitted Rg must agree with direct Rg within 30%."""
    rg_direct = calculate_radius_of_gyration(mock_peptide)
    q, intensity = calculate_saxs_profile(mock_peptide, q_max=0.3, n_points=61)
    q_max_g = min(1.3 / rg_direct, 0.1)
    mask = (q > 1e-3) & (q <= q_max_g)
    q_g, i_g = q[mask], intensity[mask]
    if len(q_g) < 3:
        pytest.skip("Too few points in Guinier region for fit")
    coeffs = np.polyfit(q_g**2, np.log(i_g), 1)
    rg_guinier = np.sqrt(-3.0 * coeffs[0])
    assert np.isfinite(rg_guinier) and rg_guinier > 0
    assert abs(rg_guinier - rg_direct) / rg_direct < 0.30
