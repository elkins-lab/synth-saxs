import os
import subprocess
import sys

import biotite.structure as struc
import numpy as np
import pytest

from synth_saxs import calculate_saxs_profile


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
    """Test CLI basic functionality."""
    # Create a dummy PDB
    pdb_path = tmp_path / "test.pdb"
    with open(pdb_path, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    out_dat = tmp_path / "out.dat"
    # Run CLI via python -m synth_saxs.cli or similar if possible,
    # but here we just test the main function directly or via subprocess.
    # Since it's not installed as a package yet in this env, we use PYTHONPATH.

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [sys.executable, "synth_saxs/cli.py", str(pdb_path), "--output", str(out_dat)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert os.path.exists(out_dat)
    data = np.loadtxt(out_dat)
    assert data.shape[0] == 51  # default n_points


def test_cli_plotting(tmp_path):
    """Test CLI plotting functionality."""
    pytest.importorskip("matplotlib")

    pdb_path = tmp_path / "test.pdb"
    with open(pdb_path, "w") as f:
        f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    plot_path = tmp_path / "plot.png"
    pr_path = tmp_path / "pr.png"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [
            sys.executable,
            "synth_saxs/cli.py",
            str(pdb_path),
            "--plot",
            str(plot_path),
            "--p-dist",
            str(pr_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert os.path.exists(plot_path)
    assert os.path.exists(pr_path)
