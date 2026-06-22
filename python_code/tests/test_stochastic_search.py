try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from StochasticSearchPy.stochastic_search import StochasticSearch


def _load_default_parameter_map():
    parameter_path = Path(__file__).resolve().parents[1] / StochasticSearch.DEFAULT_MODEL_PARAMETER_FILE
    with parameter_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = StochasticSearch._flatten_parameter_payload(payload)
    params.setdefault("N_H", params.get("N_S"))
    params.setdefault("N_sm1", params.get("N_S_m1"))
    return params


def test_sample_model_parameters_returns_single_row_dataframe(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    expected_parameters = _load_default_parameter_map()

    parameter_frame = sim.sample_model_parameters(flag_deterministic=True)

    assert isinstance(parameter_frame, pd.DataFrame)
    assert list(parameter_frame.columns) == list(sim.SAMPLED_PARAMETER_COLUMNS)
    assert len(parameter_frame.index) == 1
    assert sim.beta_H == pytest.approx(expected_parameters["beta_H"])
    assert sim.beta_M == pytest.approx(expected_parameters["beta_M"])
    assert sim.mu_H == pytest.approx(expected_parameters["mu_H"])

    row = parameter_frame.iloc[0]
    for column in sim.SAMPLED_PARAMETER_COLUMNS:
        assert row[column] == pytest.approx(getattr(sim, column))


def test_init_loads_default_model_parameters_from_json(tmp_path, sample_data_dir):
    expected_parameters = _load_default_parameter_map()
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    for key in (
        "Lambda_M",
        "Lambda_S",
        "Lambda_S_m1",
        "beta_M",
        "beta_H",
        "mu_M",
        "mu_H",
        "alpha_c",
        "alpha_h",
        "sigma",
        "p",
        "theta",
        "S_0",
        "I_10",
        "I_20",
        "M_s0",
        "M_10",
        "M_20",
        "S_m1_0",
        "Y_m1_c0",
        "Y_m1_h0",
        "R_s0",
        "R_s_m1_0",
        "Rec_0",
        "z0",
        "N_H",
        "N_sm1",
        "t0",
        "T",
        "h",
        "r_01",
        "r_02",
        "r_zero",
        "peak_df_cases",
    ):
        assert getattr(sim, key) == pytest.approx(expected_parameters[key])
    assert sim.grid_size == int(expected_parameters["grid_size"])


def test_sample_model_parameters_stochastic_path_updates_instance_state(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    np.random.seed(0)
    expected_parameters = _load_default_parameter_map()

    parameter_frame = sim.sample_model_parameters(flag_deterministic=False)

    assert isinstance(parameter_frame, pd.DataFrame)
    assert parameter_frame.loc[0, "R_s0"] == pytest.approx(0.0)
    assert parameter_frame.loc[0, "R_s_m1_0"] == pytest.approx(0.0)
    assert sim.Rec_0 == pytest.approx(0.0)
    assert sim.mu_H == pytest.approx(expected_parameters["mu_H"])
    assert sim.N_S == pytest.approx(sim.S_0 + sim.I_10 + sim.I_20 + sim.R_s0)
    assert sim.N_S_m1 == pytest.approx(sim.S_m1_0 + sim.Y_m1_c0 + sim.Y_m1_h0 + sim.R_s_m1_0)
    assert sim.Lambda_S == pytest.approx(sim.mu_H * sim.N_S)
    assert sim.Lambda_S_m1 == pytest.approx(sim.mu_H * sim.N_S_m1)
    assert sim.Lambda_M == pytest.approx(sim.mu_M * (sim.M_s0 + sim.M_10 + sim.M_20))
    assert sim.z0 == pytest.approx(sim.p * (sim.I_10 + sim.I_20 + sim.Y_m1_c0))
    assert 10.36 <= sim.b <= 33.39
    assert 0.252 <= sim.mu_M <= 0.763
    assert 0.0 < sim.beta_H <= 0.05
    assert 0.0 < sim.beta_M <= 0.05
    assert 0.581 <= sim.alpha_c <= 1.75
    assert 0.581 <= sim.alpha_h <= 1.75
    assert 0.0 < sim.sigma <= 5.0
    assert 1.0 / 60.0 <= sim.p <= 1.0 / 30.0


def test_compute_basic_reproduction_numbers_matches_baseline_parameters(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    expected_parameters = _load_default_parameter_map()

    r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()
    pi_r = (
        expected_parameters["beta_H"]
        * expected_parameters["beta_M"]
        * expected_parameters["b"] ** 2
        * expected_parameters["Lambda_M"]
    ) / (
        expected_parameters["N_H"] ** 2
        * expected_parameters["mu_M"] ** 2
    )
    expected_r01_square = pi_r * (
        (expected_parameters["N_H"] - expected_parameters["N_sm1"])
        + expected_parameters["sigma"] * (1.0 - expected_parameters["theta"]) * expected_parameters["N_sm1"]
    ) / (expected_parameters["alpha_c"] + expected_parameters["mu_H"])
    expected_r02_square = pi_r * expected_parameters["sigma"] * expected_parameters["theta"] * expected_parameters["N_sm1"] / (
        expected_parameters["alpha_c"] + expected_parameters["alpha_h"]
    )
    expected_r01 = np.sqrt(expected_r01_square)
    expected_r02 = np.sqrt(expected_r02_square)
    expected_r0 = np.sqrt(expected_r01_square + expected_r02_square)

    assert r01_per_week == pytest.approx(expected_r01)
    assert r02_per_week == pytest.approx(expected_r02)
    assert r0_per_week == pytest.approx(expected_r0)


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
    assert len(figure.axes) == 6  # Ensure the expected six subplots exist


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
    df_daily_axis = figure.axes[1]
    df_weekly_axis = figure.axes[2]
    dhf_daily_axis = figure.axes[4]
    dhf_weekly_axis = figure.axes[5]
    expected_plot_weeks = int(min(sim.T, sim.PLOT_END_WEEK) - sim.t0)
    expected_plot_days = expected_plot_weeks * sim.DAYS_PER_WEEK

    assert "7-day Average" in df_daily_axis.get_title()
    assert "3-week Average" in df_weekly_axis.get_title()
    assert "7-day Average" in dhf_daily_axis.get_title()
    assert "3-week Average" in dhf_weekly_axis.get_title()
    assert len(df_daily_axis.lines[0].get_xdata()) == expected_plot_days
    assert len(dhf_daily_axis.lines[0].get_xdata()) == expected_plot_days
    assert len(df_weekly_axis.lines[0].get_xdata()) == expected_plot_weeks
    assert len(dhf_weekly_axis.lines[0].get_xdata()) == expected_plot_weeks
    assert df_daily_axis.get_xlim()[1] == pytest.approx(sim.PLOT_END_WEEK)
    assert dhf_daily_axis.get_xlim()[1] == pytest.approx(sim.PLOT_END_WEEK)
    assert any(text.get_text().startswith("err=") for text in df_weekly_axis.texts)
    assert any(text.get_text().startswith("err=") for text in dhf_weekly_axis.texts)


def test_create_fitting_plot_includes_sampled_simulated_scatter_on_daily_and_weekly_panels(
    tmp_path,
    sample_data_dir,
):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    grid_size = int(sim.T - sim.t0) * 10000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    sim.solution = sim._build_solution_frame(
        sim.t,
        np.ones((grid_size, len(sim.ODE_STATE_COLUMNS))),
    )

    figure = sim.create_fitting_plot()

    for axis in (figure.axes[1], figure.axes[2], figure.axes[4], figure.axes[5]):
        assert len(axis.collections) >= 1


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
    df_daily_ymax = figure.axes[1].get_ylim()[1]
    df_weekly_ymax = figure.axes[2].get_ylim()[1]
    dhf_qualitative_ymax = figure.axes[3].get_ylim()[1]
    dhf_daily_ymax = figure.axes[4].get_ylim()[1]
    dhf_weekly_ymax = figure.axes[5].get_ylim()[1]

    assert df_qualitative_ymax >= 750.0
    assert dhf_qualitative_ymax >= 125.0
    assert df_daily_ymax < df_qualitative_ymax / 10.0
    assert df_weekly_ymax < df_qualitative_ymax / 5.0
    assert dhf_daily_ymax < dhf_qualitative_ymax / 4.0
    assert dhf_weekly_ymax < dhf_qualitative_ymax / 2.0


def test_compute_fitting_errors_populates_weekly_error_metrics(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    grid_size = int(sim.T - sim.t0) * 10000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    sim.solution = sim._build_solution_frame(
        sim.t,
        np.ones((grid_size, len(sim.ODE_STATE_COLUMNS))),
    )

    sim.compute_fitting_errors()

    assert isinstance(sim.df_weekly_fit_error, float)
    assert isinstance(sim.dhf_weekly_fit_error, float)
    assert np.isfinite(sim.df_weekly_fit_error)
    assert np.isfinite(sim.dhf_weekly_fit_error)


def test_create_reference_parameter_fitting_plot_uses_deterministic_reference_flow(
    tmp_path,
    sample_data_dir,
    monkeypatch,
):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    call_trace = []
    sentinel_figure = object()

    def fake_sample_model_parameters(flag_deterministic=False):
        call_trace.append(("sample_model_parameters", bool(flag_deterministic)))
        return sim._build_sample_parameter_frame()

    def fake_solve_ode_system():
        call_trace.append(("solve_ode_system", None))
        return sim.solution

    def fake_compute_fitting_errors():
        call_trace.append(("compute_fitting_errors", None))
        return None

    def fake_create_fitting_plot(figure=None):
        call_trace.append(("create_fitting_plot", figure))
        return sentinel_figure

    monkeypatch.setattr(sim, "sample_model_parameters", fake_sample_model_parameters)
    monkeypatch.setattr(sim, "solve_ode_system", fake_solve_ode_system)
    monkeypatch.setattr(sim, "compute_fitting_errors", fake_compute_fitting_errors)
    monkeypatch.setattr(sim, "create_fitting_plot", fake_create_fitting_plot)

    figure_hint = object()
    returned_figure = sim.create_reference_parameter_fitting_plot(figure=figure_hint)

    assert returned_figure is sentinel_figure
    assert call_trace == [
        ("sample_model_parameters", True),
        ("solve_ode_system", None),
        ("compute_fitting_errors", None),
        ("create_fitting_plot", figure_hint),
    ]


def test_comparison_frames_split_plot_and_error_intervals(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    time_weeks = np.array([25.0, 39.0, 53.0], dtype=np.float64)
    solution_values = np.ones((len(time_weeks), len(sim.ODE_STATE_COLUMNS)), dtype=np.float64)
    sim.solution = sim._build_solution_frame(time_weeks, solution_values)

    df_error_frame, dhf_error_frame = sim._build_fitting_comparison_frames()
    df_plot_frame, dhf_plot_frame = sim._build_plot_comparison_frames()

    expected_plot_start_date = pd.Timestamp(datetime.date.fromisocalendar(2010, 25, 1))
    expected_error_start_date = pd.Timestamp(datetime.date.fromisocalendar(2010, 37, 1))
    assert df_plot_frame["time_week"].iloc[0] == pytest.approx(time_weeks[0])
    assert dhf_plot_frame["time_week"].iloc[0] == pytest.approx(time_weeks[0])
    assert df_plot_frame.index[0] == expected_plot_start_date
    assert dhf_plot_frame.index[0] == expected_plot_start_date
    assert df_error_frame["time_week"].iloc[0] == pytest.approx(sim.FITTING_ERROR_START_WEEK)
    assert dhf_error_frame["time_week"].iloc[0] == pytest.approx(sim.FITTING_ERROR_START_WEEK)
    assert df_error_frame.index[0].strftime("%Y-%m-%d") == "2010-09-13"
    assert dhf_error_frame.index[0].strftime("%Y-%m-%d") == "2010-09-13"
    assert df_error_frame.index[0] == expected_error_start_date
    assert dhf_error_frame.index[0] == expected_error_start_date
    assert len(df_error_frame.index) == sim.FITTING_WINDOW_WEEKS * sim.DAYS_PER_WEEK
    assert len(dhf_error_frame.index) == sim.FITTING_WINDOW_WEEKS * sim.DAYS_PER_WEEK


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


def test_save_fitting_scale_solution_writes_plot_ready_csv(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    grid_size = int(sim.T - sim.t0) * 10000
    sim.t = np.linspace(sim.t0, sim.T, grid_size)
    solution_values = np.ones((grid_size, len(sim.ODE_STATE_COLUMNS)), dtype=np.float64)
    solution_values[:, sim.ODE_STATE_COLUMNS.index("z")] = np.linspace(0.0, 140.0, grid_size)
    solution_values[:, sim.ODE_STATE_COLUMNS.index("Y_m1_h")] = np.linspace(10.0, 80.0, grid_size)
    sim.solution = sim._build_solution_frame(sim.t, solution_values)

    output_path = sim.save_fitting_scale_solution(
        sim.output_dir / "solution_fitting_scale.csv"
    )

    fitting_scale = pd.read_csv(output_path)
    assert list(fitting_scale.columns) == [
        "frequency",
        "outcome",
        "date",
        "time_week",
        "observed_count",
        "simulated_count",
        "observed_moving_average",
        "simulated_moving_average",
        "moving_average_window",
    ]
    assert set(fitting_scale["frequency"]) == {"daily", "weekly"}
    assert set(fitting_scale["outcome"]) == {"DF", "DHF"}
    daily = fitting_scale[fitting_scale["frequency"] == "daily"]
    weekly = fitting_scale[fitting_scale["frequency"] == "weekly"]
    assert set(daily["moving_average_window"]) == {sim.MOVING_AVERAGE_WINDOW_DAYS}
    assert set(weekly["moving_average_window"]) == {sim.WEEKLY_MOVING_AVERAGE_WINDOW_WEEKS}
    assert daily["date"].min() == "2010-07-05"
    assert daily["simulated_count"].notna().all()


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
    fitting_scale_solution_path = sim.runtime_dir / snapshot["artifacts"]["fitting_scale_solution"]
    solution_time_series_path = sim.runtime_dir / snapshot["artifacts"]["solution_time_series"]
    assert snapshot["sample_index"] == 3
    assert snapshot["acceptance_thresholds"]["fitting_window_weeks"] == 10
    assert snapshot["acceptance_thresholds"]["df_fitting_start_index"] == 10
    assert snapshot["acceptance_thresholds"]["dhf_fitting_start_index"] == 6
    assert snapshot["acceptance_thresholds"]["fitting_start_week"] == pytest.approx(37.0)
    assert snapshot["acceptance_thresholds"]["fitting_end_week"] == pytest.approx(47.0)
    assert snapshot["acceptance_thresholds"]["fitting_start_date"] == "2010-09-13"
    assert parameter_path.exists()
    assert solution_path.exists()
    assert fitting_scale_solution_path.exists()
    assert solution_time_series_path.exists()
    raw_solution = pd.read_csv(solution_path)
    assert list(raw_solution.columns) == list(sim.ODE_SOLUTION_COLUMNS)
    fitting_scale_solution = pd.read_csv(fitting_scale_solution_path)
    assert {"daily", "weekly"}.issubset(set(fitting_scale_solution["frequency"]))

    reloaded = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "reloaded")
    reloaded.load_parameters_from_json(parameter_path)
    reloaded.load_solution_snapshot(solution_path)
    reloaded.compute_fitting_errors()
    reloaded.df_error_threshold = snapshot["acceptance_thresholds"]["df_fit_error_max"]
    reloaded.dhf_error_threshold = snapshot["acceptance_thresholds"]["dhf_fit_error_max"]
    r01_per_week, r02_per_week, r0_per_week = reloaded.compute_basic_reproduction_numbers()

    assert reloaded.df_fit_error == pytest.approx(snapshot["acceptance_metrics"]["df_fit_error"])
    assert reloaded.dhf_fit_error == pytest.approx(snapshot["acceptance_metrics"]["dhf_fit_error"])
    assert reloaded.df_weekly_fit_error == pytest.approx(snapshot["acceptance_metrics"]["df_weekly_fit_error"])
    assert reloaded.dhf_weekly_fit_error == pytest.approx(snapshot["acceptance_metrics"]["dhf_weekly_fit_error"])
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

    state = np.concatenate((state_vector, state_host, state_host_m1, [0.0]), dtype=float)
    derivatives = StochasticSearch.compute_ode_rhs(state, 0.0, **params)

    vector_conservation_law = np.sum(derivatives[0:3])
    host_conservation_law = np.sum(derivatives[3:7])
    host_m1_conservation_law= np.sum(derivatives[7:11])
    assert abs(vector_conservation_law) < 1e-8
    assert abs(host_conservation_law) < 1e-8
    assert abs(host_m1_conservation_law) < 1e-8
    assert np.isfinite(derivatives[-1])
