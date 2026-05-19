import numpy as np
import biotite.structure as struc
import pytest
import os
from synth_saxs import get_form_factor, calculate_saxs_profile, calculate_radius_of_gyration

class TestSAXSRigor:
    """Scientific rigor tests for SAXS simulation based on peer-reviewed standards."""

    def test_atomic_form_factors_at_zero_q(self) -> None:
        """Verify form factors converge to atomic number Z at q=0."""
        q_zero = np.array([0.0])
        benchmarks = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}

        for elem, z_expected in benchmarks.items():
            f_0 = get_form_factor(elem, q_zero)[0]
            assert np.abs(f_0 - z_expected) < 0.1

    def test_ubiquitin_rg_internal_consistency(self) -> None:
        """Verify that Rg from scattering curve matches Rg from coordinates."""
        pdb_path = "tests/data/1UBQ.pdb"
        if not os.path.exists(pdb_path):
            pytest.skip("1UBQ.pdb test data not found.")

        import biotite.structure.io.pdb as pdb_io
        pdb_file = pdb_io.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1)
        structure = structure[(structure.chain_id == "A") & (~structure.hetero)]

        rg_coord = calculate_radius_of_gyration(structure)
        q_max_guinier = 1.3 / rg_coord
        q, intensity = calculate_saxs_profile(
            structure, q_min=0.0, q_max=q_max_guinier, n_points=50, include_solvent=False
        )

        q2 = q**2
        ln_i = np.log(intensity)
        slope, _ = np.polyfit(q2, ln_i, 1)
        rg_estimated = np.sqrt(-3 * slope)

        assert np.abs(rg_estimated - rg_coord) / rg_coord < 0.02

    def test_kratky_folding_signature(self) -> None:
        """Verify Kratky plot distinguishes between folded and disordered states."""
        # 1. Folded State: Compact sphere
        n_atoms = 100
        struct_folded = struc.AtomArray(n_atoms)
        struct_folded.coord = np.zeros((n_atoms, 3))
        np.random.seed(42)
        r = np.random.uniform(0, 10, n_atoms)
        theta = np.random.uniform(0, np.pi, n_atoms)
        phi = np.random.uniform(0, 2*np.pi, n_atoms)
        struct_folded.coord[:, 0] = r * np.sin(theta) * np.cos(phi)
        struct_folded.coord[:, 1] = r * np.sin(theta) * np.sin(phi)
        struct_folded.coord[:, 2] = r * np.cos(theta)
        struct_folded.element = ["C"] * n_atoms

        # 2. Disordered State: Two atoms very far apart (limit case of expansion)
        struct_disordered = struc.AtomArray(2)
        struct_disordered.coord = np.array([[0, 0, 0], [100, 0, 0]])
        struct_disordered.element = ["C", "C"]

        q = np.linspace(0.01, 0.5, 50)
        _, i_folded = calculate_saxs_profile(struct_folded, q_min=0.01, q_max=0.5, n_points=50, include_solvent=False)
        _, i_disordered = calculate_saxs_profile(struct_disordered, q_min=0.01, q_max=0.5, n_points=50, include_solvent=False)

        k_folded = (q**2) * i_folded
        k_disordered = (q**2) * i_disordered

        # Folded peak check: should decay at high q relative to its own peak
        peak_idx_f = np.argmax(k_folded)
        assert k_folded[-1] < k_folded[peak_idx_f]

        # Disordered Kratky should rise (or plateau) relative to its start
        assert k_disordered[-1] > k_disordered[0]
