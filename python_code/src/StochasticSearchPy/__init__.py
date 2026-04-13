"""Public package API for the Two Strains Dengue HMS model.

The package exposes three main entry points:

- :class:`StochasticSearch` for model simulation and search workflows
- :class:`DataProcessing` for dataset preparation and tabulation helpers
- :func:`copy_reference_dataset` for copying the committed reference dataset

Example
-------
>>> from StochasticSearchPy import StochasticSearch
>>> model = StochasticSearch()
"""

from .data_assets import DATA_DIR_ENV_VAR, copy_reference_dataset
from .data_processing import DataProcessing
from .stochastic_search import StochasticSearch

__all__ = [
    "DATA_DIR_ENV_VAR",
    "DataProcessing",
    "StochasticSearch",
    "copy_reference_dataset",
]
