import biotite.structure as struc
import numpy as np
import pytest

from synth_saxs.engine import SaxsSimulator, add_noise, calculate_p_dist, calculate_saxs_profile
from synth_saxs.visualization import plot_p_dist, plot_saxs_results


def test_calculate_saxs_profile_empty():
    """Verify that an empty structure returns zeros gracefully."""
    atoms = struc.AtomArray(0)
    atoms.coord = np.zeros((0, 3))
    atoms.element = []

    q, intensity = calculate_saxs_profile(atoms, n_points=10)
    assert len(q) == 10
    assert np.all(intensity == 0)


def test_calculate_saxs_profile_rejects_invalid_q_grid():
    """Verify invalid q-grid inputs fail with actionable errors."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.zeros((1, 3))
    atoms.element = ["C"]

    with pytest.raises(ValueError, match="n_points"):
        calculate_saxs_profile(atoms, n_points=0)
    with pytest.raises(ValueError, match="n_points"):
        calculate_saxs_profile(atoms, n_points=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        calculate_saxs_profile(atoms, q_min="low")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        calculate_saxs_profile(atoms, q_min=np.nan)
    with pytest.raises(ValueError, match="q_max"):
        calculate_saxs_profile(atoms, q_min=0.5, q_max=0.1)


def test_calculate_saxs_profile_rejects_nonfinite_coordinates():
    """Verify SAXS calculation rejects NaN/inf coordinates before distance calculation."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[np.nan, 0.0, 0.0]])
    atoms.element = ["C"]

    with pytest.raises(ValueError, match="coordinates must be finite"):
        calculate_saxs_profile(atoms)


def test_calculate_p_dist_too_few_atoms():
    """Verify that P(r) handles < 2 atoms without crashing."""
    # 0 atoms
    atoms0 = struc.AtomArray(0)
    atoms0.coord = np.zeros((0, 3))
    r0, p0 = calculate_p_dist(atoms0, bins=5)
    assert np.all(p0 == 0)

    # 1 atom
    atoms1 = struc.AtomArray(1)
    atoms1.coord = np.zeros((1, 3))
    atoms1.element = ["C"]
    r1, p1 = calculate_p_dist(atoms1, bins=5)
    assert np.all(p1 == 0)


def test_calculate_p_dist_rejects_invalid_inputs():
    """Verify invalid P(r) inputs fail before histogram calculation."""
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [10, 0, 0]])
    atoms.element = ["C", "C"]

    with pytest.raises(ValueError, match="bins"):
        calculate_p_dist(atoms, bins=0)
    with pytest.raises(ValueError, match="bins"):
        calculate_p_dist(atoms, bins=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        calculate_p_dist(atoms, r_max="far")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="r_max"):
        calculate_p_dist(atoms, r_max=0)


def test_calculate_p_dist_custom_rmax():
    """Verify that P(r) respects a custom r_max."""
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [10, 0, 0]])
    atoms.element = ["C", "C"]

    # Case 1: r_max is exactly the distance
    r, p = calculate_p_dist(atoms, bins=10, r_max=10.0)
    assert r[-1] < 10.0

    # Case 2: r_max is larger than distance
    r_large, p_large = calculate_p_dist(atoms, bins=20, r_max=20.0)
    assert r_large[-1] < 20.0
    # Peak should be in the middle (bin representing ~10A)
    assert np.argmax(p_large) == 9 or np.argmax(p_large) == 10


def test_add_noise_zero_level():
    """Verify add_noise with zero noise level."""
    intensity = np.array([1.0, 2.0, 3.0])
    noisy = add_noise(intensity, noise_level=0.0)
    assert np.allclose(noisy, intensity)


def test_add_noise_clamping():
    """Verify that noise doesn't result in negative intensities."""
    intensity = np.array([1e-12, 1e-12])
    # Extreme noise that would definitely go negative if not clamped
    noisy = add_noise(intensity, noise_level=100.0)
    assert np.all(noisy >= 1e-10)


def test_saxs_simulator_list_input():
    """Verify SaxsSimulator handles lists of AtomArrays."""
    atoms1 = struc.AtomArray(1)
    atoms1.coord = np.zeros((1, 3))
    atoms1.element = ["C"]

    atoms2 = struc.AtomArray(1)
    atoms2.coord = np.array([[5, 5, 5]])
    atoms2.element = ["C"]

    sim = SaxsSimulator(n_points=5)
    intensity = sim.simulate([atoms1, atoms2])
    assert len(intensity) == 5
    assert np.all(intensity > 0)


def test_saxs_simulator_empty_list():
    """Verify SaxsSimulator handles empty list."""
    sim = SaxsSimulator(n_points=5)
    intensity = sim.simulate([])
    assert len(intensity) == 5
    assert np.all(intensity == 0)


def test_calculate_saxs_profile_solvent_density():
    """Verify that solvent density affects the profile."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.zeros((1, 3))
    atoms.element = ["C"]

    # Higher density should lead to more subtraction (lower I(0))
    _, i_low = calculate_saxs_profile(atoms, solvent_density=0.1)
    _, i_high = calculate_saxs_profile(atoms, solvent_density=0.3)

    assert i_high[0] < i_low[0]


def test_visualization_missing_rg_guinier():
    """Verify Guinier plot handles missing Rg by using default mask."""
    pytest.importorskip("matplotlib")

    q = np.linspace(0.01, 0.5, 50)
    i = np.exp(-(q**2) * 10)
    fig = plot_saxs_results(q, i, plot_type="guinier", rg=None)
    assert fig is not None


def test_plot_p_dist_no_matplotlib_manual():
    """Manually verify plot_p_dist when HAS_MATPLOTLIB is False (mocked)."""
    import synth_saxs.visualization

    with pytest.MonkeyPatch().context() as m:
        m.setattr(synth_saxs.visualization, "HAS_MATPLOTLIB", False)
        res = plot_p_dist(np.array([1]), np.array([1]))
        assert res is None
