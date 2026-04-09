import pytest

from StochasticSearchPy import StochasticSearch


def test_stochastic_search_uses_runtime_artifact_directory(tmp_path, sample_data_dir):
    runtime_dir = tmp_path / "runtime"
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=runtime_dir)

    assert sim.runtime_dir == runtime_dir
    assert sim.plots_dir == runtime_dir / "plots"
    assert sim.output_dir == runtime_dir / "parameters"
    assert sim.frecuency_per_week_DF.shape == (2, 2)
    assert sim.frecuency_per_week_DHF.shape == (2, 2)


def test_compute_r_zero_matches_baseline_parameters(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")

    r01_per_week, r02_per_week, r0_per_week = sim.compute_r_zero()

    assert r01_per_week == pytest.approx(1.7783839818240441)
    assert r02_per_week == pytest.approx(0.15248169655310542)
    assert r0_per_week == pytest.approx(1.7849090325817885)


def test_update_conditions_search_requires_all_thresholds(tmp_path, sample_data_dir):
    sim = StochasticSearch(data_dir=sample_data_dir, runtime_dir=tmp_path / "runtime")
    sim.r_zero = 1.1
    sim.fitting_error_DF = sim.bound_error_FD - 1
    sim.fitting_error_DHF = sim.bound_error_FHD - 1
    sim.z_max = 699

    assert sim.update_conditions_search() is True

    sim.z_max = 701
    assert sim.update_conditions_search() is False
