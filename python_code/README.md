# Two Strains Dengue HMS Model

This repository now exposes the dengue search code as an installable Python package under a standard `src/` layout and adds the minimum scaffolding needed for a cleaner CI/CD and test-driven workflow.

## What changed

- `StochasticSearchPy` remains the importable package.
- The reference dataset now lives under `data/reference/raw_data/` instead of inside the Python package.
- Runtime artifacts now default to `./artifacts/` instead of writing into the package directory.
- A console entry point is available through `two-strains-dengue`.
- `pyproject.toml` defines install, test, and build metadata.
- `tests/` provides a small regression suite around stable model behavior.
- `docs/` provides package-level API and workflow documentation.
- `mkdocs.yml` configures the documentation site that GitHub Pages deploys.
- repository-level workflows under `../.github/workflows/` run CI and GitHub Pages.

## Recommended layout

```text
python_code/
├── .github/workflows/ci.yml
├── README.md
├── pyproject.toml
├── docs/
├── src/
│   └── StochasticSearchPy/
│       ├── __init__.py
│       ├── cli.py
│       ├── data_assets.py
│       ├── data_processing.py
│       ├── stochastic_search.py
├── data/
│   └── reference/
│       └── raw_data/  # committed reference dataset, not bundled in the wheel
├── tests/
└── artifacts/  # created at runtime, ignored by git
```

## Local development

Create an environment and install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

If you want to run tests without installing first, the suite already adds `src/` to `sys.path`.

Run the tests:

```bash
python -m pytest
```

Build the package:

```bash
python -m build
```

Prepare a working data directory from the committed reference dataset:

```bash
two-strains-dengue prepare-data --output-dir ./artifacts/data
```

Run a smoke check from the CLI:

```bash
two-strains-dengue smoke --data-dir ./artifacts/data --runtime-dir ./artifacts
```

Generate frequency tables:

```bash
two-strains-dengue frequency-tables --data-dir ./path/to/data
```

Run the stochastic search:

```bash
two-strains-dengue search --data-dir ./artifacts/data --runtime-dir ./artifacts --samples 1
```

## Data management

The package no longer bundles the 64 MB reference dataset inside the wheel. That keeps builds and installs smaller and makes the runtime data dependency explicit.

- Source checkouts can prepare a working copy with `two-strains-dengue prepare-data --output-dir ./artifacts/data`.
- Installed environments can point to an existing prepared directory with `--data-dir`.
- You can also set `TWO_STRAINS_DENGUE_DATA_DIR` to a prepared directory and omit `--data-dir`.
- The committed reference source used by `prepare-data` lives at `./data/reference/raw_data`.

## TDD workflow

Use this order for future changes:

1. Add or update a focused test in `tests/`.
2. Make the smallest production code change needed to satisfy that test.
3. Run `python -m pytest`.
4. Run `python -m build` before merging so packaging regressions are caught early.

The current tests are intentionally lightweight and filesystem-isolated so they can run in CI without depending on large local datasets.

## Additional docs

- Package overview: `docs/index.md`
- API summary: `docs/api.md`
- Development workflow: `docs/workflows.md`
- GitHub and Pages setup: `docs/github-pages.md`
