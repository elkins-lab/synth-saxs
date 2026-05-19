# Publishing synth-saxs to PyPI

This guide explains how to publish `synth-saxs` to PyPI.

## Prerequisites

1.  **Install Build Tools**:
    ```bash
    pip install build twine
    ```

2.  **PyPI Account**: Ensure you have an account at [PyPI](https://pypi.org/).

## Publishing Workflow

### 1. Build the Package
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build wheels and source dist
python -m build
```

### 2. Upload to PyPI (Manual)
```bash
python -m twine upload dist/*
```

## Automated Publishing (GitHub Actions)

The repository is configured to automatically publish to PyPI when a **GitHub Release** is created.

1.  Add your PyPI API token to GitHub Secrets as `PYPI_API_TOKEN`.
2.  Update the version in `pyproject.toml`.
3.  Create and publish a new Release on GitHub.

## Versioning
Follow Semantic Versioning (`MAJOR.MINOR.PATCH`).
Update the version in `pyproject.toml`:
```toml
[project]
version = "0.1.1"
```
