# API Reference

## Public package imports

```python
from StochasticSearchPy import (
    DATA_DIR_ENV_VAR,
    DataProcessing,
    StochasticSearch,
    copy_reference_dataset,
)
```

## `DataProcessing`

Primary responsibilities:

- resolve the active data directory
- create and query the SQLite-backed incidence tables
- aggregate raw case rows into weekly DF and DHF frequency tables

Common methods:

- `DataProcessing(data_dir=None)`: open the configured dataset
- `build_daily_frequency_tables()`: return per-date DF/DHF frequency tables and save them as CSV
- `build_weekly_frequency_tables()`: return per-week DF/DHF frequency tables and save them as CSV
- `build_weekly_frequency_arrays()`: return weekly DF/DHF frequency tables as NumPy arrays
- `export_incidence_rows_with_weeks()`: export per-case incidence rows with their epidemiological weeks
- `close_database_connection()`: release the SQLite connection

## `StochasticSearch`

Primary responsibilities:

- load prepared model inputs
- integrate the compartmental ODE system
- evaluate fit/error thresholds
- write plots and parameter snapshots to a runtime directory

Common methods:

- `StochasticSearch(data_dir=None, runtime_dir=None)`: initialize the model
- `solve_ode_system()`: run the ODE solver and return a `pandas.DataFrame`
  with `time_grid` plus one column per compartment state
- `compute_fitting_errors()`: compute DF/DHF fit errors
- `compute_basic_reproduction_numbers()`: compute the reproduction number components
- `save_solution_plots()` and `save_fitting_plot()`: write plots
- `save_parameter_snapshot(file_name_prefix=None)`: write a YAML parameter snapshot

## Data helpers

- `copy_reference_dataset(output_dir, source_dir=None)`: copy the committed
  reference dataset into a writable working directory
- `DATA_DIR_ENV_VAR`: environment variable name,
  `TWO_STRAINS_DENGUE_DATA_DIR`

## CLI commands

The package installs the `two-strains-dengue` command.

- `two-strains-dengue prepare-data --output-dir ./artifacts/data`
- `two-strains-dengue smoke --data-dir ./artifacts/data --runtime-dir ./artifacts`
- `two-strains-dengue frequency-tables --data-dir ./artifacts/data`
- `two-strains-dengue search --data-dir ./artifacts/data --runtime-dir ./artifacts --samples 1`
