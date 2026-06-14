import biotite.structure as struc
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from synth_saxs import calculate_saxs_profile


@given(
    coords=arrays(np.float64, (5, 3), elements=st.floats(min_value=-100, max_value=100)),
    translation=arrays(np.float64, (3,), elements=st.floats(min_value=-50, max_value=50)),
)
def test_saxs_translation_invariance(coords, translation):
    """Verify that SAXS profile is invariant to translation."""
    atoms = struc.AtomArray(5)
    atoms.coord = coords
    atoms.element = ["C"] * 5

    q1, i1 = calculate_saxs_profile(atoms, n_points=5, include_solvent=False)

    atoms.coord += translation
    q2, i2 = calculate_saxs_profile(atoms, n_points=5, include_solvent=False)

    assert np.allclose(i1, i2, atol=1e-5)


@given(coords=arrays(np.float64, (5, 3), elements=st.floats(min_value=-10, max_value=10)))
def test_saxs_rotation_invariance(coords):
    """Verify that SAXS profile is invariant to rotation."""
    atoms = struc.AtomArray(5)
    atoms.coord = coords
    atoms.element = ["C"] * 5

    q1, i1 = calculate_saxs_profile(atoms, n_points=5, include_solvent=False)

    # Random rotation matrix
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    atoms.coord = np.dot(atoms.coord, R.T)
    q2, i2 = calculate_saxs_profile(atoms, n_points=5, include_solvent=False)

    assert np.allclose(i1, i2, atol=1e-5)
