import json
from pathlib import Path

NOTEBOOK_DIR = Path("examples/interactive_tutorials")


def _notebook_source(path: Path) -> str:
    data = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )


def test_saxs_profile_tutorial_debye_equation_renders():
    """Verify the Debye equation markdown has a valid LaTeX fraction."""
    source = _notebook_source(NOTEBOOK_DIR / "saxs_profile_generation.ipynb")

    assert "\f" not in source
    assert "\\frac{\\sin(q r_{ij})}{q r_{ij}}" in source


def test_end_to_end_tutorial_installs_imported_synth_packages():
    """Verify the Colab install cell includes the synth-suite packages imported later."""
    source = _notebook_source(NOTEBOOK_DIR / "end_to_end_validation.ipynb")

    for package_name in ["synth-nmr", "synth-afm", "synth-cryo-em"]:
        assert package_name in source


def test_hydration_tutorial_caches_bulk_profile():
    """Verify the slider callback does not recompute the invariant bulk profile."""
    source = _notebook_source(NOTEBOOK_DIR / "hydration_shell_analysis.ipynb")

    assert "q_bulk, i_bulk = calculate_saxs_profile" in source
    assert "q = q_bulk" in source
