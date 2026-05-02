try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

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
