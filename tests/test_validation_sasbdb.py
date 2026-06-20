import os

import biotite.structure.io.pdb as pdb_io
import numpy as np
import pytest

from synth_saxs import calculate_saxs_profile


def _load_sasbdb_dat(path):
    """Robust parser for SASBDB .dat files."""
    exp_data = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                row = [float(x) for x in parts]
                if len(row) >= 2:
                    exp_data.append(row)
            except ValueError:
                continue
    return np.array(exp_data)


def _validate_against_sasbdb(pdb_code, sasbdb_code, q_max_fit=0.3):
    """
    Helper to validate a PDB model against a SASBDB experimental dataset.
    """
    data_path = f"tests/data/{sasbdb_code}.dat"
    pdb_path = f"tests/data/{pdb_code}.pdb"

    if not os.path.exists(data_path) or not os.path.exists(pdb_path):
        import pytest

        pytest.skip(f"Data for {sasbdb_code} or {pdb_code} not found.")

    # 1. Load Experimental Data
    exp_data = _load_sasbdb_dat(data_path)
    q_exp_nm = exp_data[:, 0]
    i_exp = exp_data[:, 1]

    # Convert q from nm^-1 to A^-1 (1 nm^-1 = 0.1 A^-1)
    q_exp_a = q_exp_nm / 10.0

    # 2. Load PDB and Calculate Synthetic Profile
    pdb_file = pdb_io.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)
    # Filter for protein atoms only
    protein = structure[~structure.hetero]

    # Calculate synthetic profile
    _, i_synth = calculate_saxs_profile(
        protein,
        q_min=q_exp_a.min(),
        q_max=q_exp_a.max(),
        n_points=len(q_exp_a),
        hydration_shell_density=0.03,
    )

    # 3. Assessment: Correlation Coefficient
    # Focus on scientifically relevant range (q < q_max_fit)
    mask = (q_exp_a < q_max_fit) & (i_exp > 0) & (i_synth > 0)
    i_exp_fit = i_exp[mask]
    i_synth_fit = i_synth[mask]

    # Robust scaling in log space
    log_scale = np.mean(np.log(i_exp_fit) - np.log(i_synth_fit))
    scale_factor = np.exp(log_scale)
    i_synth_scaled = i_synth_fit * scale_factor

    # Correlation of log-intensities
    # RATIONALE: SAXS data varies over orders of magnitude; log-scale correlation
    # is the standard for assessing agreement in curve shape.
    correlation = np.corrcoef(np.log(i_exp_fit), np.log(i_synth_scaled))[0, 1]

    print(f"\n  Validation {sasbdb_code} ({pdb_code}) for q < {q_max_fit} A^-1:")
    print(f"  Correlation Coefficient (log scale): {correlation:.4f}")

    # Threshold: 0.97 is a high standard for real-world experimental data
    # (accounting for noise, buffer subtraction artifacts, and static PDB vs solution ensemble).
    assert correlation > 0.97, f"Correlation for {sasbdb_code} too low: {correlation:.4f}"
    return correlation


def test_sasbdb_ubiquitin_validation():
    """Validate against Ubiquitin (SASDAQ2)."""
    _validate_against_sasbdb("1UBQ", "SASDAQ2")


def test_sasbdb_lysozyme_validation():
    """Validate against Lysozyme (SASDAB2)."""
    # Lysozyme is larger and data might go higher
    _validate_against_sasbdb("1AKI", "SASDAB2")


def test_sasbdb_bsa_validation():
    """
    Validate against Bovine Serum Albumin (SASDBT4).
    BSA is a 66 kDa monomeric standard highly cited in SAXS literature.
    """
    # The PDB and DAT are both saved under SASDBT4
    _validate_against_sasbdb("SASDBT4", "SASDBT4")


@pytest.mark.skip(reason="Test is too slow for standard validation suite.")
def test_sasbdb_myoglobin_validation():
    """
    Validate against Myoglobin (SASDA92).
    Myoglobin (~17 kDa) is an established monomeric standard in structural biology and SAXS.
    """
    _validate_against_sasbdb("SASDA92", "SASDA92")


if __name__ == "__main__":
    test_sasbdb_ubiquitin_validation()
    test_sasbdb_lysozyme_validation()
    test_sasbdb_bsa_validation()
    test_sasbdb_myoglobin_validation()
