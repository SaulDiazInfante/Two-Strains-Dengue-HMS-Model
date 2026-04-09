# API Reference

## Public package imports

```python
from StochasticSearchPy import (
    DATA_DIR_ENV_VAR,
    DataProcessing,
    StochasticSearch,
    prepare_data_directory,
)
```

## `DataProcessing`

Primary responsibilities:

- resolve the active data directory
- create and query the SQLite-backed incidence tables
- aggregate raw case rows into weekly DF and DHF frequency tables

Common methods:

- `DataProcessing(data_dir=None)`: open the configured dataset
- `incidence_frequency_tables()`: return weekly DF/DHF tables as pandas DataFrames
- `frecuency_per_day_and_week()`: generate the legacy `.dat` files used by the model
- `close()`: release the SQLite connection

## `StochasticSearch`

Primary responsibilities:

- load prepared model inputs
- integrate the compartmental ODE system
- evaluate fit/error thresholds
- write plots and parameter snapshots to a runtime directory

Common methods:

- `StochasticSearch(data_dir=None, runtime_dir=None)`: initialize the model
- `ode_int_solution()`: run the ODE solver
- `fitting_error()`: compute DF/DHF fit errors
- `compute_r_zero()`: compute the reproduction number components
- `solution_plot()` and `fitting_plot()`: write plots
- `save_parameters(path_prefix=None)`: write a YAML parameter snapshot

## Data helpers

- `prepare_data_directory(output_dir, source_dir=None)`: copy the committed
  reference dataset into a writable working directory
- `DATA_DIR_ENV_VAR`: environment variable name,
  `TWO_STRAINS_DENGUE_DATA_DIR`

## CLI commands

The package installs the `two-strains-dengue` command.

- `two-strains-dengue prepare-data --output-dir ./artifacts/data`
- `two-strains-dengue smoke --data-dir ./artifacts/data --runtime-dir ./artifacts`
- `two-strains-dengue frequency-tables --data-dir ./artifacts/data`
- `two-strains-dengue search --data-dir ./artifacts/data --runtime-dir ./artifacts --samples 1`
