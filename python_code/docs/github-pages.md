# GitHub Setup

## Repository layout

This repository uses GitHub-facing configuration at the repository root and keeps
the Python package under `python_code/`.

- GitHub Actions workflows live in `.github/workflows/`
- package code lives in `python_code/src/StochasticSearchPy/`
- package documentation source lives in `python_code/docs/`
- MkDocs configuration lives in `python_code/mkdocs.yml`

## CI

The CI workflow should run from the repository root and execute package commands
inside `python_code/`. The current workflow is designed to verify:

- installation in editable mode
- unit tests
- reference data preparation
- a CLI smoke run
- package build

## GitHub Pages

The Pages workflow should:

1. install the docs dependencies from `python_code/`
2. build the site with `mkdocs build --strict`
3. upload the generated `python_code/site/` directory
4. deploy the artifact with the official GitHub Pages deployment action

Expected site URL:

- `https://sauldiazinfante.github.io/Two-Strains-Dengue-HMS-Model/`

## Manual repository setting

In the GitHub repository settings:

1. Open `Settings`.
2. Open `Pages`.
3. Set the build and deployment source to `GitHub Actions`.

After that, pushes to the default branch should publish the docs site through the
Pages workflow.
