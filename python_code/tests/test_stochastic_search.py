try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

import json

import numpy as np
import pandas as pd
import pytest

from StochasticSearchPy.stochastic_search import StochasticSearch


def test_sample_model_parameters_returns_single_row_dataframe(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    parameter_frame = sim.sample_model_parameters(flag_deterministic=True)

    assert isinstance(parameter_frame, pd.DataFrame)
    assert list(parameter_frame.columns) == list(sim.SAMPLED_PARAMETER_COLUMNS)
    assert len(parameter_frame.index) == 1

    row = parameter_frame.iloc[0]
    for column in sim.SAMPLED_PARAMETER_COLUMNS:
        assert row[column] == pytest.approx(getattr(sim, column))


def test_sample_model_parameters_stochastic_path_updates_instance_state(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    np.random.seed(0)

    parameter_frame = sim.sample_model_parameters(flag_deterministic=False)

    assert isinstance(parameter_frame, pd.DataFrame)
    assert parameter_frame.loc[0, "R_s0"] == pytest.approx(0.0)
    assert parameter_frame.loc[0, "R_s_m1_0"] == pytest.approx(0.0)
    assert sim.Rec_0 == pytest.approx(0.0)
    assert sim.mu_H == pytest.approx(0.000039 * 7.0)


def test_compute_basic_reproduction_numbers_matches_baseline_parameters(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()

    assert r01_per_week == pytest.approx(1.9033635638379547)
    assert r02_per_week == pytest.approx(0.17132774893607353)
    assert r0_per_week == pytest.approx(1.9110588828451682)


def test_create_fitting_plot(tmp_path, sample_data_dir):
    """
    Test that create_fitting_plot generates a matplotlib figure.
    This requires inputs to be initialized within the StochasticSearch instance.
    """
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    # Prepare necessary attributes to mock a completed simulation.
    # t and solution must match the real grid: linspace(t0=25, T=53, grid_size=280000)
    # and the solution dataframe must expose named DHF and reported-DF columns.
    grid_size = int(sim.T - sim.t0) * 10000  # 280000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    sim.solution = sim._build_solution_frame(
        sim.t,
        np.ones((grid_size, len(sim.ODE_STATE_COLUMNS))),
    )
    sim.weekly_df_frequency_array = np.array([[0, 10], [1, 12], [2, 14], [3, 16]])
    sim.weekly_dhf_frequency_array = np.array([[0, 5], [1, 8], [2, 11]])

    figure = sim.create_fitting_plot()

    # Validate that the returned object is a matplotlib figure
    assert figure is not None
    assert figure.get_axes()  # Ensure there are axes present
    assert len(figure.axes) == 4  # Ensure the expected four subplots exist


def test_create_fitting_plot_uses_moving_average_fitting_window(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    grid_size = int(sim.T - sim.t0) * 10000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    sim.solution = sim._build_solution_frame(
        sim.t,
        np.ones((grid_size, len(sim.ODE_STATE_COLUMNS))),
    )
    sim.weekly_df_frequency_array = np.array([[0, 10], [1, 12], [2, 14], [3, 16]])
    sim.weekly_dhf_frequency_array = np.array([[0, 5], [1, 8], [2, 11]])
    sim.compute_fitting_errors()

    figure = sim.create_fitting_plot()
    df_fitting_axis = figure.axes[1]
    dhf_fitting_axis = figure.axes[3]

    assert "7-day average" in df_fitting_axis.get_title()
    assert "7-day average" in dhf_fitting_axis.get_title()
    assert len(df_fitting_axis.lines[0].get_xdata()) == sim.FITTING_WINDOW_WEEKS * sim.DAYS_PER_WEEK
    assert len(dhf_fitting_axis.lines[0].get_xdata()) == sim.FITTING_WINDOW_WEEKS * sim.DAYS_PER_WEEK


def test_create_fitting_plot_uses_observation_scale_on_fitting_panels_only(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    grid_size = int(sim.T - sim.t0) * 10000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    solution_values = np.ones((grid_size, len(sim.ODE_STATE_COLUMNS)))
    z_index = sim.ODE_STATE_COLUMNS.index("z")
    y_m1_h_index = sim.ODE_STATE_COLUMNS.index("Y_m1_h")
    solution_values[12345, z_index] = 750.0
    solution_values[23456, y_m1_h_index] = 125.0
    sim.solution = sim._build_solution_frame(sim.t, solution_values)
    sim.weekly_df_frequency_array = np.array([[0, 10], [1, 12], [2, 14], [3, 16]])
    sim.weekly_dhf_frequency_array = np.array([[0, 5], [1, 8], [2, 11]])

    figure = sim.create_fitting_plot()
    df_qualitative_ymax = figure.axes[0].get_ylim()[1]
    df_fitting_ymax = figure.axes[1].get_ylim()[1]
    dhf_qualitative_ymax = figure.axes[2].get_ylim()[1]
    dhf_fitting_ymax = figure.axes[3].get_ylim()[1]

    assert df_qualitative_ymax >= 750.0
    assert dhf_qualitative_ymax >= 125.0
    assert df_fitting_ymax < 25.0
    assert dhf_fitting_ymax < 15.0


def test_solve_ode_system_replaces_zero_placeholder_solution(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    sim.grid_size = 250
    sim.h = np.float64(sim.T) / np.float64(sim.grid_size)

    solution = sim.solve_ode_system()

    assert isinstance(solution, pd.DataFrame)
    assert list(solution.columns) == list(sim.ODE_SOLUTION_COLUMNS)
    assert solution.shape == (sim.grid_size, len(sim.ODE_SOLUTION_COLUMNS))
    assert np.isfinite(solution.to_numpy()).all()
    assert np.allclose(solution[sim.TIME_GRID_COLUMN].to_numpy(), sim.t)
    assert not np.allclose(solution[list(sim.ODE_STATE_COLUMNS)].to_numpy(), 0.0)
    assert solution is sim.solution


def test_save_sampled_solution_time_series_uses_daily_time_scale(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    time_weeks = np.array([25.0, 26.0, 27.0], dtype=np.float64)
    solution_values = np.zeros((len(time_weeks), len(sim.ODE_STATE_COLUMNS)), dtype=np.float64)
    z_index = sim.ODE_STATE_COLUMNS.index("z")
    y_m1_h_index = sim.ODE_STATE_COLUMNS.index("Y_m1_h")
    solution_values[:, z_index] = np.array([0.0, 7.0, 14.0], dtype=np.float64)
    solution_values[:, y_m1_h_index] = np.array([14.0, 7.0, 0.0], dtype=np.float64)
    sim.solution = sim._build_solution_frame(time_weeks, solution_values)

    output_path = sim.save_sampled_solution_time_series(
        sim.output_dir / "solution_time_series.csv"
    )

    sampled = pd.read_csv(output_path)
    assert list(sampled.columns) == [sim.TIME_DAY_COLUMN, *sim.ODE_SOLUTION_COLUMNS]
    assert sampled.shape[0] == 15
    assert sampled.loc[0, sim.TIME_DAY_COLUMN] == 0
    assert sampled.loc[sampled.index[-1], sim.TIME_DAY_COLUMN] == 14
    assert sampled.loc[0, sim.TIME_GRID_COLUMN] == pytest.approx(25.0)
    assert sampled.loc[sampled.index[-1], sim.TIME_GRID_COLUMN] == pytest.approx(27.0)
    assert np.allclose(np.diff(sampled[sim.TIME_DAY_COLUMN].to_numpy()), 1.0)
    assert np.allclose(
        np.diff(sampled[sim.TIME_GRID_COLUMN].to_numpy()),
        np.full(sampled.shape[0] - 1, 1.0 / sim.DAYS_PER_WEEK),
    )
    assert sampled.loc[7, "z"] == pytest.approx(7.0)
    assert sampled.loc[7, "Y_m1_h"] == pytest.approx(7.0)


def test_save_acceptance_snapshot_writes_reproducible_artifacts(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    sim.grid_size = 250
    sim.h = np.float64(sim.T) / np.float64(sim.grid_size)
    sim.df_error_threshold = 1.0e9
    sim.dhf_error_threshold = 1.0e9

    sim.solve_ode_system()
    sim.compute_fitting_errors()
    sim.compute_basic_reproduction_numbers()
    sim.evaluate_search_acceptance()

    snapshot_path = sim.save_acceptance_snapshot(
        sample_index=3,
        snapshot_timestamp="20260503T120000",
    )

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    parameter_path = sim.runtime_dir / snapshot["artifacts"]["parameter_snapshot"]
    solution_path = sim.runtime_dir / snapshot["artifacts"]["solution_snapshot"]
    solution_time_series_path = sim.runtime_dir / snapshot["artifacts"]["solution_time_series"]
    assert snapshot["sample_index"] == 3
    assert parameter_path.exists()
    assert solution_path.exists()
    assert solution_time_series_path.exists()

    reloaded = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "reloaded")
    reloaded.load_parameters_from_json(parameter_path)
    reloaded.load_solution_snapshot(solution_path)
    reloaded.compute_fitting_errors()
    r01_per_week, r02_per_week, r0_per_week = reloaded.compute_basic_reproduction_numbers()

    assert reloaded.df_fit_error == pytest.approx(snapshot["acceptance_metrics"]["df_fit_error"])
    assert reloaded.dhf_fit_error == pytest.approx(snapshot["acceptance_metrics"]["dhf_fit_error"])
    assert r01_per_week == pytest.approx(snapshot["acceptance_metrics"]["r01_per_week"])
    assert r02_per_week == pytest.approx(snapshot["acceptance_metrics"]["r02_per_week"])
    assert r0_per_week == pytest.approx(snapshot["acceptance_metrics"]["r0_per_week"])
    assert reloaded.evaluate_search_acceptance() == snapshot["acceptance_metrics"]["meets_acceptance_criteria"]
    sampled_solution = pd.read_csv(solution_time_series_path)
    assert sim.TIME_DAY_COLUMN in sampled_solution.columns
    assert sampled_solution.loc[0, sim.TIME_DAY_COLUMN] == 0


def test_evaluate_search_acceptance_requires_all_thresholds(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    sim.r_zero = 1.1
    sim.df_fit_error = sim.df_error_threshold - 1
    sim.dhf_fit_error = sim.dhf_error_threshold - 1
    sim.peak_df_cases = 699

    assert sim.evaluate_search_acceptance() is True

    sim.peak_df_cases = 701
    assert sim.evaluate_search_acceptance() is False


def test_compute_ode_rhs():
    state_vector = np.array([10000.0, 500.0, 700.0])
    state_host = np.array([18000.0, 1.0, 1.0, 0.0])
    state_host_m1 = np.array([6000.0, 1.0, 1.0, 0.0])
    n_vector = state_vector.sum()
    n_host = state_host.sum()
    n_host_m1 = state_host_m1.sum()

    mu_M = 0.08333333333333333
    mu_H = 3.913894324853229e-05

    Lambda_M = n_vector * mu_M
    Lambda_S = n_host * mu_H
    Lambda_S_m1 = n_host_m1 * mu_H

    beta_M = 0.4
    beta_H = 0.3
    b = 2.0

    alpha_c = 0.2
    alpha_h = 0.4
    sigma = 1.5
    p = 0.25
    theta = 0.2

    params = {
        'Lambda_M': Lambda_M,
        'Lambda_S': Lambda_S,
        'Lambda_S_m1': Lambda_S_m1,
        'beta_M': beta_M,
        'beta_H': beta_H,
        'b': b,
        'mu_M': mu_M,
        'mu_H': mu_H,
        'alpha_c': alpha_c,
        'alpha_h': alpha_h,
        'sigma': sigma,
        'p': p,
        'theta': theta,
        'n_vector': n_vector,
        'n_host': n_host,
        'n_host_m1': n_host_m1
    }

    state = np.concatenate((state_vector, state_host, state_host_m1), dtype=float)
    derivatives = StochasticSearch.compute_ode_rhs(state, 0.0, **params)

    vector_conservation_law = np.sum(derivatives[0:3])
    host_conservation_law = np.sum(derivatives[3:7])
    host_m1_conservation_law= np.sum(derivatives[7:11])
    assert abs(vector_conservation_law) < 1e-8
    assert abs(host_conservation_law) < 1e-8
    assert abs(host_m1_conservation_law) < 1e-8
