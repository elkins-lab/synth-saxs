import numpy as np

from synth_saxs.fitting import (
    calculate_chi_squared,
    fit_profile,
    interpolate_profile,
)


def test_interpolate_profile() -> None:
    q_calc = np.linspace(0.01, 0.5, 50)
    i_calc = np.exp(-10 * q_calc**2)

    q_exp = np.array([0.05, 0.1, 0.2, 0.3])
    i_exp_interp = interpolate_profile(q_exp, q_calc, i_calc)

    # Expected analytical values
    expected = np.exp(-10 * q_exp**2)
    np.testing.assert_allclose(i_exp_interp, expected, rtol=1e-3)


def test_fit_profile() -> None:
    # Construct a mock experiment where theoretical is exactly scaled and shifted
    q_exp = np.linspace(0.01, 0.5, 10)
    i_calc_interp = np.exp(-10 * q_exp**2)

    c_true = 2.5
    k_true = 0.01
    i_exp = c_true * i_calc_interp + k_true
    err_exp = np.ones_like(i_exp) * 0.001

    c_fit, k_fit, chi_sq = fit_profile(i_exp, err_exp, i_calc_interp)

    assert np.isclose(c_fit, c_true)
    assert np.isclose(k_fit, k_true)
    assert np.isclose(chi_sq, 0.0, atol=1e-5)


def test_calculate_chi_squared() -> None:
    i_exp = np.array([10.0, 5.0, 1.0])
    i_fit = np.array([9.0, 5.0, 2.0])
    err_exp = np.array([1.0, 0.5, 0.5])

    # residuals: (10-9)/1 = 1, (5-5)/0.5 = 0, (1-2)/0.5 = -2
    # sum of squares = 1^2 + 0^2 + (-2)^2 = 1 + 4 = 5
    # dof = 3 points - 2 params = 1
    # reduced chi_sq = 5 / 1 = 5.0

    chi_sq = calculate_chi_squared(i_exp, err_exp, i_fit, n_params=2)
    assert chi_sq == 5.0
