try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

import numpy as np
import pytest
from pathlib import Path

from StochasticSearchPy import StochasticSearch


def test_stochastic_search_uses_runtime_artifact_directory(tmp_path, sample_data_dir):
    runtime_dir = tmp_path / "runtime"
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=runtime_dir)

    assert sim.runtime_dir == runtime_dir
    assert sim.plots_dir == runtime_dir / "plots"
    assert sim.output_dir == runtime_dir / "parameters"
    assert sim.weekly_df_frequency_array.shape == (2, 2)
    assert sim.weekly_dhf_frequency_array.shape == (2, 2)
    assert Path(sample_data_dir / "frequency_per_week_DF.csv").exists()
    assert Path(sample_data_dir / "frequency_per_week_DHF.csv").exists()


def test_compute_basic_reproduction_numbers_matches_baseline_parameters(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()

    assert r01_per_week == pytest.approx(1.7783839818240441)
    assert r02_per_week == pytest.approx(0.15248169655310542)
    assert r0_per_week == pytest.approx(1.7849090325817885)


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
    state_vector = np.array([10000.0, 500.0, 700.0] )
    state_host = np.array([18000.0, 1.0, 1.0, 0.0])
    state_host_m1 = np.array([6000.0, 1.0, 1.0, 0.0])
    n_vector = state_vector.sum()
    n_host = state_host.sum()
    n_host_m1 = state_host_m1.sum()

    mu_M = 0.08333333333333333
    mu_H = 3.913894324853229e-05

    Lambda_M = n_vector * mu_M
    Lambda_S = n_host * mu_H
    Lambda_S_m1 = n_host_m1 *  mu_H

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

