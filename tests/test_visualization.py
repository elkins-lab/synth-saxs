import os

import numpy as np
import pytest

from synth_saxs import plot_saxs_results


def test_plot_saxs_results_standard(tmp_path):
    """Test standard SAXS plot."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    output_path = str(tmp_path / "saxs_std.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="standard")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_kratky(tmp_path):
    """Test Kratky plot."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    output_path = str(tmp_path / "saxs_kratky.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="kratky")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_guinier(tmp_path):
    """Test Guinier plot and Rg estimation logic."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.001, 0.1, 50)
    # I(q) = I(0) * exp(-q^2 * Rg^2 / 3)
    rg_target = 20.0
    intensity = 100 * np.exp(-(q**2) * (rg_target**2) / 3.0)

    output_path = str(tmp_path / "saxs_guinier.png")
    fig = plot_saxs_results(
        q, intensity, output_path=output_path, plot_type="guinier", rg=rg_target
    )
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_no_matplotlib():
    """Verify graceful failure when matplotlib is missing."""
    with pytest.MonkeyPatch().context() as m:
        import synth_saxs.visualization

        m.setattr(synth_saxs.visualization, "HAS_MATPLOTLIB", False)
        fig = plot_saxs_results(np.array([0.1]), np.array([1.0]))
        assert fig is None


def test_plot_saxs_results_porod(tmp_path):
    """Test Porod plot."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = 1.0 / (q**4)

    output_path = str(tmp_path / "saxs_porod.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="porod")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_all(tmp_path):
    """Test 'all' plot type."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    output_path = str(tmp_path / "saxs_all.png")
    fig = plot_saxs_results(q, intensity, output_path=output_path, plot_type="all")
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_p_dist(tmp_path):
    """Test P(r) plot."""
    from synth_saxs.visualization import plot_p_dist

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    r = np.linspace(0, 100, 50)
    p_r = r**2 * np.exp(-(r**2) / 1000)

    output_path = str(tmp_path / "p_dist.png")
    fig = plot_p_dist(r, p_r, output_path=output_path)
    assert fig is not None
    assert os.path.exists(output_path)


def test_plot_saxs_results_no_output():
    """Test SAXS plot without saving to file."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    fig = plot_saxs_results(q, intensity, output_path=None)
    assert fig is not None


def test_plot_p_dist_no_output():
    """Test P(r) plot without saving to file."""
    from synth_saxs.visualization import plot_p_dist

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    r = np.linspace(0, 100, 50)
    p_r = r**2 * np.exp(-(r**2) / 1000)

    fig = plot_p_dist(r, p_r, output_path=None)
    assert fig is not None


def test_plot_saxs_results_invalid_type():
    """Test SAXS plot with an invalid plot type (should just return fig with standard plot)."""
    # Note: argparse should prevent this in CLI, but internal API should be robust.
    q = np.linspace(0.01, 0.5, 50)
    intensity = np.exp(-(q**2) * 100)

    # This shouldn't crash
    fig = plot_saxs_results(q, intensity, plot_type="invalid_type")
    assert fig is not None


def test_guinier_positive_slope():
    """Verify Guinier fit handles non-physical positive slope (imaginary Rg)."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    q = np.linspace(0.01, 0.1, 10)
    # Intensity that INCREASES with q (non-physical for Guinier)
    intensity = np.exp(q**2 * 10)

    fig = plot_saxs_results(q, intensity, plot_type="guinier")
    assert fig is not None
    # Code should handle this by taking max(0, -3*slope) -> Rg=0
    # We just ensure it doesn't crash.


def test_matplotlib_import_failure():
    """Attempt to cover the ImportError block in visualization.py using reload."""
    import sys
    from importlib import reload
    from unittest.mock import patch

    # Mock matplotlib to be missing during reload
    with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
        import synth_saxs.visualization

        reload(synth_saxs.visualization)
        assert synth_saxs.visualization.HAS_MATPLOTLIB is False

    # Restore state for other tests
    reload(synth_saxs.visualization)
