import os
from unittest.mock import patch

import biotite.structure as struc
import numpy as np
import pytest

from synth_saxs import calculate_saxs_profile
from synth_saxs.cli import main


def test_hydration_shell_effect():
    """Verify that hydration shell density increases the intensity (more contrast)."""
    atoms = struc.AtomArray(1)
    atoms.coord = np.zeros((1, 3))
    atoms.element = ["C"]

    # Bulk density is subtracted.
    # f_eff = f_vac - (rho_sol - rho_shell) * V * decay
    # If rho_shell > 0, (rho_sol - rho_shell) is smaller, so subtraction is smaller, I(0) is larger.
    _, i_bulk = calculate_saxs_profile(atoms, hydration_shell_density=0.0)
    _, i_shell = calculate_saxs_profile(atoms, hydration_shell_density=0.03)

    assert i_shell[0] > i_bulk[0]


def test_cli_basic(tmp_path):
    """Test CLI basic functionality by calling main() directly."""
    # Create a dummy PDB
    pdb_path = tmp_path / "test.pdb"
    with open(pdb_path, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    out_dat = tmp_path / "out.dat"
    test_args = ["synth-saxs", str(pdb_path), "--output", str(out_dat)]

    with patch("sys.argv", test_args):
        main()

    assert os.path.exists(out_dat)
    data = np.loadtxt(out_dat)
    assert data.shape[0] == 51  # default n_points


def test_cli_plotting(tmp_path):
    """Test CLI plotting functionality by calling main() directly."""
    pytest.importorskip("matplotlib")

    pdb_path = tmp_path / "test.pdb"
    with open(pdb_path, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    plot_path = tmp_path / "plot.png"
    pr_path = tmp_path / "pr.png"
    pr_dat = tmp_path / "pr.dat"

    test_args = [
        "synth-saxs",
        str(pdb_path),
        "--plot",
        str(plot_path),
        "--p-dist",
        str(pr_path),
        "--p-dist-dat",
        str(pr_dat),
    ]

    with patch("sys.argv", test_args):
        main()

    assert os.path.exists(plot_path)
    assert os.path.exists(pr_path)
    assert os.path.exists(pr_dat)


def test_cli_error_handling(tmp_path):
    """Test CLI error handling with non-existent file."""
    test_args = ["synth-saxs", "non_existent.pdb"]

    with patch("sys.argv", test_args), patch("sys.exit") as mock_exit:
        main()
        mock_exit.assert_called_with(1)


def test_cli_multi_model(tmp_path):
    """Test CLI handling of multi-model PDB files."""
    pdb_path = tmp_path / "multi.pdb"
    with open(pdb_path, "w") as f:
        # Model 1
        f.write("MODEL        1\n")
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        f.write("ENDMDL\n")
        # Model 2
        f.write("MODEL        2\n")
        f.write("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00  0.00           C\n")
        f.write("ENDMDL\n")

    test_args = ["synth-saxs", str(pdb_path)]
    with patch("sys.argv", test_args):
        # Just ensure it doesn't crash and picks model 1
        main()
