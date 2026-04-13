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
