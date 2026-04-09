# Two Strains Dengue HMS Model

[![CI](https://github.com/SaulDiazInfante/Two-Strains-Dengue-HMS-Model/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/SaulDiazInfante/Two-Strains-Dengue-HMS-Model/actions/workflows/ci.yml)

This repository contains the packaged Python implementation of the Two Strains
Dengue HMS model. The Python package, tests, and documentation source live in
`python_code/`, while repository-level GitHub automation lives in `.github/`.

## Quick links

- Package source: [`python_code/src/StochasticSearchPy/`](python_code/src/StochasticSearchPy)
- Package README: [`python_code/README.md`](python_code/README.md)
- Documentation source: [`python_code/docs/`](python_code/docs)
- GitHub Pages site: https://sauldiazinfante.github.io/Two-Strains-Dengue-HMS-Model/

## Local development

```bash
cd python_code
python -m pip install -e ".[dev,docs]"
python -m pytest
mkdocs build --strict
```

## Data preparation and smoke test

```bash
cd python_code
two-strains-dengue prepare-data --output-dir ./artifacts/data
two-strains-dengue smoke --data-dir ./artifacts/data --runtime-dir ./artifacts
```

## GitHub automation

- `.github/workflows/ci.yml` runs tests, docs build, data prep, smoke checks, and package build.
- `.github/workflows/pages.yml` builds and deploys the MkDocs site to GitHub Pages.

To activate GitHub Pages deployment, set the repository Pages source to `GitHub Actions` in the repository settings.
