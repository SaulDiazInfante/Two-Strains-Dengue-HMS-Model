try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

import inspect

from StochasticSearchPy import DataProcessing, StochasticSearch, copy_reference_dataset
from StochasticSearchPy import cli, data_assets


def test_public_modules_have_docstrings():
    assert inspect.getdoc(cli)
    assert inspect.getdoc(data_assets)


def test_public_api_has_docstrings():
    documented_objects = [
        DataProcessing,
        DataProcessing.close_database_connection,
        DataProcessing.build_daily_frequency_tables,
        DataProcessing.build_weekly_frequency_tables,
        DataProcessing.build_weekly_frequency_arrays,
        StochasticSearch,
        StochasticSearch.compute_basic_reproduction_numbers,
        StochasticSearch.solve_ode_system,
        StochasticSearch.sample_model_parameters,
        copy_reference_dataset,
        cli.build_cli_parser,
        cli.run_prepare_data_command,
        cli.run_search_command,
        cli.run_cli,
    ]

    for obj in documented_objects:
        assert inspect.getdoc(obj), f"Missing docstring for {obj}"
