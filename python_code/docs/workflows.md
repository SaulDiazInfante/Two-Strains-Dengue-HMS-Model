# Workflows

## Data preparation

The package does not bundle the committed reference dataset inside the wheel.
For local or CI runs, create a working copy first:

```bash
two-strains-dengue prepare-data --output-dir ./artifacts/data
```

You can then either:

- pass `--data-dir ./artifacts/data` to CLI commands, or
- set `TWO_STRAINS_DENGUE_DATA_DIR=./artifacts/data`

The committed reference source lives under `data/reference/raw_data/`.

## Local development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

## Recommended TDD loop

1. Add or update a focused test in `tests/`.
2. Add or refine docstrings if you changed the public surface.
3. Implement the smallest production change that satisfies the test.
4. Run `python -m pytest`.
5. Run `python -m build` before merging.

## CI/CD expectations

The CI workflow currently checks:

- install in editable mode
- unit tests
- data preparation
- a smoke run of the CLI
- package build

The GitHub Pages workflow builds and deploys the documentation site from
`python_code/docs/` with MkDocs.

Repository-level workflows live in `.github/workflows/` at the git repository
root, not inside `python_code/`.
