import inspect

from StochasticSearchPy import DataProcessing, StochasticSearch, prepare_data_directory
from StochasticSearchPy import cli, data_assets


def test_public_modules_have_docstrings():
    assert inspect.getdoc(cli)
    assert inspect.getdoc(data_assets)


def test_public_api_has_docstrings():
    documented_objects = [
        DataProcessing,
        DataProcessing.close,
        DataProcessing.incidence_frequency_tables,
        DataProcessing.frecuency_per_day_and_week,
        StochasticSearch,
        StochasticSearch.compute_r_zero,
        StochasticSearch.ode_int_solution,
        StochasticSearch.parameters_sampling,
        prepare_data_directory,
        cli.build_parser,
        cli.run_prepare_data,
        cli.run_search,
        cli.main,
    ]

    for obj in documented_objects:
        assert inspect.getdoc(obj), f"Missing docstring for {obj}"
