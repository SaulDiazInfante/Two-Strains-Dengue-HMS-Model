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

    M_s, M_1, M_2 = state_vector
    S, I_1, I_2, R_s = state_host
    S_m1, Y_m1_c, Y_m1_h, R_s_m1 = state_host_m1

    total_host_population = (Lambda_S + Lambda_S_m1) / mu_H
    host_to_vector_force = beta_M * b / total_host_population
    vector_to_host_force = beta_H * b / total_host_population
    A_I1 = host_to_vector_force * I_1
    A_I2 = host_to_vector_force * I_2
    A_Y_m1_c = host_to_vector_force * Y_m1_c
    A_Y_m1_h = host_to_vector_force * Y_m1_h
    B_M1 = vector_to_host_force * M_1
    B_M2 = vector_to_host_force * M_2
    reinfection_incidence = sigma * B_M2 * S_m1

    dM_s = Lambda_M - (A_I1 + A_I2 + A_Y_m1_c + A_Y_m1_h) * M_s - mu_M * M_s
    dM_1 = A_I1 * M_s - mu_M * M_1
    dM_2 = (A_I2 + A_Y_m1_c + A_Y_m1_h) * M_s - mu_M * M_2
    expected_vector_conservation_law = (dM_s + dM_1 + dM_2)
    assert abs(expected_vector_conservation_law) < 1e-8

    dS = Lambda_S - (B_M1 + B_M2) * S - mu_H * S
    dI_1 = B_M1 * S - (alpha_c + mu_H) * I_1
    dI_2 = B_M2 * S - (alpha_c + mu_H) * I_2
    dR_s = alpha_c * (I_1 + I_2)  - mu_H * R_s
    expected_host_conservation_law = (dS + dI_1 + dI_2 + dR_s)
    assert abs(expected_host_conservation_law) < 1e-8

    dS_m1 = Lambda_S_m1 - (reinfection_incidence + mu_H * S_m1)
    dY_m1_c = (1.0 - theta) * reinfection_incidence - (alpha_c + mu_H) * Y_m1_c
    dY_m1_h = theta * reinfection_incidence - (alpha_h + mu_H) * Y_m1_h
    dR_s_m1 = alpha_c * Y_m1_c + alpha_h * Y_m1_h  - mu_H * R_s_m1
    expected_host_m1_conservation_law = ( dS_m1 + dY_m1_c + dY_m1_h + dR_s_m1 )
    assert abs(expected_host_m1_conservation_law) < 1e-8

    dZ = p * (I_1 + I_2 + Y_m1_c)

    state = np.concatenate((state_vector, state_host, state_host_m1), dtype=float)
    derivatives = StochasticSearch.compute_ode_rhs(state, 0.0, **params)
    expected = np.array(
        [   dM_s, dM_1, dM_2,
            dS, dI_1, dI_2, dR_s,
            dS_m1, dY_m1_c, dY_m1_h, dR_s_m1
        ]
    )
    vector_conservation_law = np.sum(expected[0:3])
    host_conservation_law = np.sum(expected[3:7])
    host_m1_conservation_law= np.sum(expected[7:11])
    assert abs(vector_conservation_law) < 1e-8
    assert abs(host_conservation_law) < 1e-8
    assert abs(host_m1_conservation_law) < 1e-8
    assert derivatives == pytest.approx(expected)
