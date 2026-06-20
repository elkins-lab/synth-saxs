import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack

from synth_saxs.cli import main
from synth_saxs.engine import preprocess_structure
from synth_saxs.fitting import (
    calculate_chi_squared,
    fit_profile,
    load_experimental_data,
)
from synth_saxs.visualization import plot_saxs_results


def test_load_experimental_data_3col() -> None:
    """Test loading a standard 3-column data file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("# q I err\n")
        f.write("0.01 100.0 1.0\n")
        f.write("0.02 50.0 0.5\n")
        f.write("0.03 -10.0 0.1\n")  # Negative intensity, should be filtered
        filepath = f.name

    try:
        q, i, err = load_experimental_data(filepath)
        assert len(q) == 2
        assert len(i) == 2
        assert len(err) == 2
        assert np.allclose(q, [0.01, 0.02])
    finally:
        os.remove(filepath)


def test_load_experimental_data_2col() -> None:
    """Test loading a 2-column data file to trigger the fallback branch."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("0.01 100.0\n")
        f.write("0.02 50.0\n")
        filepath = f.name

    try:
        q, i, err = load_experimental_data(filepath)
        assert len(q) == 2
        assert np.allclose(err, [1.0, 1.0])
    finally:
        os.remove(filepath)


def test_load_experimental_data_errors() -> None:
    """Test error handling in load_experimental_data."""
    # Unparseable data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("not a number\n")
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="Could not parse"):
            load_experimental_data(filepath)
    finally:
        os.remove(filepath)

    # 1 column data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("0.01\n")
        filepath = f.name

    try:
        with pytest.raises(ValueError, match="at least 2 columns"):
            load_experimental_data(filepath)
    finally:
        os.remove(filepath)


def test_fit_profile_edge_cases() -> None:
    """Test edge cases in fit_profile and calculate_chi_squared."""
    # Not enough valid points
    i_exp = np.array([100.0, 50.0])
    err_exp = np.array([1.0, 0.5])
    i_calc = np.array([100.0, 50.0])

    with pytest.raises(ValueError, match="Not enough overlapping"):
        fit_profile(i_exp, err_exp, i_calc)

    # Chi-squared degrees of freedom <= 0
    # calculate_chi_squared takes n_params=2 by default
    chi_sq = calculate_chi_squared(np.array([1.0, 2.0]), np.array([0.1, 0.1]), np.array([1.0, 2.0]))
    assert np.isinf(chi_sq)


def test_visualization_exp_data() -> None:
    """Test visualization paths with exp_data overlay."""
    import matplotlib.pyplot as plt

    q = np.linspace(0.01, 0.3, 100)
    i = np.exp(-10 * q**2)

    # Dummy exp data
    q_exp = np.linspace(0.01, 0.3, 20)
    i_exp = np.exp(-10 * q_exp**2) + np.random.normal(0, 0.05, 20)
    err_exp = np.ones_like(i_exp) * 0.05
    c, k = 1.0, 0.0
    exp_data = (q_exp, i_exp, err_exp, c, k)

    # Should hit lines 73-77
    fig_std = plot_saxs_results(q, i, plot_type="standard", exp_data=exp_data)
    assert fig_std is not None

    # Should hit grid plotting with exp_data (lines 65-70 and 139-173)
    fig_all = plot_saxs_results(q, i, plot_type="all", exp_data=exp_data)
    assert fig_all is not None

    # Clean up plots
    plt.close("all")


def test_cli_fit_argument() -> None:
    """Test the CLI logic for the --fit argument."""
    # Create a dummy PDB file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(
            "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N  \n"
            "ATOM      2  CA  ALA A   1      10.000  11.500  10.000  1.00  0.00           C  \n"
            "ATOM      3  C   ALA A   1      11.500  11.500  10.000  1.00  0.00           C  \n"
            "ATOM      4  O   ALA A   1      12.000  10.500  10.000  1.00  0.00           O  \n"
        )
        pdb_path = f.name

    # Create dummy experimental data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("0.01 100.0 1.0\n0.02 50.0 0.5\n0.03 25.0 0.25\n")
        dat_path = f.name

    test_args = [
        "synth-saxs",
        pdb_path,
        "--fit",
        dat_path,
    ]

    # Patch sys.argv to simulate CLI call, and patch plt.show so it doesn't block
    with patch("sys.argv", test_args):
        with patch("matplotlib.pyplot.show"):
            main()

    os.remove(pdb_path)
    os.remove(dat_path)


def test_engine_atom_array_stack() -> None:
    """Test preprocess_structure with an AtomArrayStack."""
    import biotite.structure as struc

    array1 = AtomArray(4)
    array1.res_name = np.array(["ALA", "PEG", "GLY", "CL"])
    array1.coord = np.random.rand(4, 3)

    array2 = AtomArray(4)
    array2.res_name = np.array(["ALA", "PEG", "GLY", "CL"])
    array2.coord = np.random.rand(4, 3)

    stack = struc.stack([array1, array2])

    filtered = preprocess_structure(stack)
    assert isinstance(filtered, AtomArrayStack)
    assert filtered.array_length() == 2
    assert list(filtered.res_name) == ["ALA", "GLY"]
