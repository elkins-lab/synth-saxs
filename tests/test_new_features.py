import os
import numpy as np
import pytest
import biotite.structure as struc
from synth_saxs.engine import calculate_p_dist, add_noise, get_form_factor
from synth_saxs.visualization import plot_saxs_results, plot_p_dist

def test_p_dist_calculation():
    """Verify P(r) calculation for a simple 2-atom system."""
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [10, 0, 0]])
    atoms.element = ["C", "C"]
    
    r, p_r = calculate_p_dist(atoms, bins=10)
    
    assert len(r) == 10
    assert len(p_r) == 10
    # There should be exactly one peak around r=10
    assert np.argmax(p_r) == 9 # Last bin if r_max is ~10.2
    assert p_r[9] > 0
    assert np.sum(p_r[:8]) == 0

def test_add_noise():
    """Verify that noise is added and respects bounds."""
    intensity = np.array([100.0, 50.0, 10.0])
    noisy = add_noise(intensity, noise_level=0.1)
    
    assert noisy.shape == intensity.shape
    assert not np.array_equal(noisy, intensity)
    assert np.all(noisy > 0)

def test_new_form_factors():
    """Verify that newly added elements are accessible."""
    q = np.array([0.1])
    for elem in ["NA", "CL", "MG", "FE", "ZN"]:
        f = get_form_factor(elem, q)
        assert f[0] > 0
        # Should not be equal to Carbon (fallback)
        f_c = get_form_factor("C", q)
        assert not np.allclose(f, f_c)

def test_porod_plot(tmp_path):
    """Verify Porod plot generation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
        
    q = np.linspace(0.1, 0.5, 20)
    intensity = q**-4 # Perfect Porod decay
    
    output_path = str(tmp_path / "porod.png")
    fig = plot_saxs_results(q, intensity, plot_type="porod", output_path=output_path)
    assert fig is not None
    assert os.path.exists(output_path)

def test_all_plots_2x2(tmp_path):
    """Verify 'all' plot type with 2x2 grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
        
    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-q**2 * 10)
    
    output_path = str(tmp_path / "all_plots.png")
    fig = plot_saxs_results(q, intensity, plot_type="all", output_path=output_path, rg=15.0)
    assert fig is not None
    assert len(fig.axes) == 4
    assert os.path.exists(output_path)

def test_p_dist_plot(tmp_path):
    """Verify P(r) plot generation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
        
    r = np.linspace(0, 50, 50)
    p_r = np.exp(-(r-25)**2 / 100)
    
    output_path = str(tmp_path / "p_dist.png")
    fig = plot_p_dist(r, p_r, output_path=output_path)
    assert fig is not None
    assert os.path.exists(output_path)

def test_p_dist_calculation_stack():
    """Verify P(r) calculation with AtomArrayStack."""
    stack = struc.AtomArrayStack(1, 2)
    stack.coord = np.array([[[0, 0, 0], [10, 0, 0]]])
    stack.element = ["C", "C"]
    
    # Passing stack[0] would be an AtomArray, but let's pass the stack itself 
    # to hit the ndim == 3 check if biotite allows it or we cast it.
    from typing import Any, cast
    r, p_r = calculate_p_dist(cast(Any, stack))
    assert len(r) == 50 # default bins
    assert np.max(p_r) > 0

def test_guinier_fit_no_rg():
    """Verify Guinier fit without provided Rg."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
        
    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-q**2 * 10)
    fig = plot_saxs_results(q, intensity, plot_type="guinier")
    assert len(fig.axes) == 1

def test_guinier_fit_insufficient_points():
    """Verify Guinier fit handles insufficient points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
        
    q = np.array([0.01])
    intensity = np.array([1.0])
    fig = plot_saxs_results(q, intensity, plot_type="guinier")
    # Should have drawn the error text
    assert len(fig.axes) == 1
