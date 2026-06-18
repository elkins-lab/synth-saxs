import os

import pytest

# Set Matplotlib backend to 'Agg' to avoid TclError on headless CI environments (Windows)
os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture(autouse=True)
def skip_on_missing_matplotlib():
    """Fixture to skip tests if matplotlib is not installed but requested."""
    pass
