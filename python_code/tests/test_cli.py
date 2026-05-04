try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

from pathlib import Path

import matplotlib.pyplot as plt

from StochasticSearchPy import data_assets
from StochasticSearchPy import cli as cli_module
from StochasticSearchPy import run_cli as package_run_cli
from StochasticSearchPy.cli import run_cli


def test_run_cli_is_importable_from_package_root():
    assert package_run_cli is run_cli


def test_smoke_command_runs_with_explicit_data_dir(tmp_path, sample_data_dir, capsys):
    runtime_dir = tmp_path / "runtime"

    exit_code = run_cli(
        [
            "smoke",
            "--data-dir",
            str(sample_data_dir),
            "--runtime-dir",
            str(runtime_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "r0=" in captured.out


def test_frequency_tables_command_writes_output(sample_data_dir, capsys):
    exit_code = run_cli(["frequency-tables", "--data-dir", str(sample_data_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "generated=" in captured.out


def test_prepare_data_command_copies_reference_files(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "reference"
    output_dir = tmp_path / "prepared"
    source_dir.mkdir()
    (source_dir / "alpha.csv").write_text("alpha", encoding="utf-8")
    monkeypatch.setattr(data_assets, "REFERENCE_DATA_FILES", ("alpha.csv",))

    exit_code = run_cli(
        [
            "prepare-data",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "copied_files=1" in captured.out
    assert (output_dir / "alpha.csv").exists()


def test_search_command_updates_timestamped_fitting_plot(tmp_path, monkeypatch, capsys):
    class FakeSearch:
        figure_ids = []

        def __init__(self, data_dir=None, runtime_dir=None):
            self.data_dir = Path(data_dir) if data_dir is not None else None
            self.runtime_dir = Path(runtime_dir)
            self.plots_dir = self.runtime_dir / "plots"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = self.runtime_dir / "parameters"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.sample_count = 0
            self.df_error_threshold = 100.0
            self.dhf_error_threshold = 25.0
            self.df_fit_error = 0.0
            self.dhf_fit_error = 0.0
            self.peak_df_cases = 0.0
            self.acceptance_call_count = 0

        def build_weekly_frequency_tables(self):
            return None

        def sample_model_parameters(self, flag_deterministic=False):
            return None

        def solve_ode_system(self):
            return None

        def compute_fitting_errors(self):
            self.df_fit_error = 11.0
            self.dhf_fit_error = 7.0
            self.peak_df_cases = 123.0

        def compute_basic_reproduction_numbers(self):
            return 1.1, 1.2, 1.3

        def evaluate_search_acceptance(self):
            self.acceptance_call_count += 1
            return self.acceptance_call_count == 2

        def create_fitting_plot(self, figure=None):
            if figure is None:
                figure = plt.figure()
            else:
                figure.clear()
            axis = figure.add_subplot(111)
            axis.plot([0.0, 1.0], [0.0, 1.0])
            self.figure_ids.append(id(figure))
            return figure

        def save_solution_plots(self):
            (self.plots_dir / "populations_grid.png").write_text("plot", encoding="utf-8")
            return None

        def save_acceptance_snapshot(
            self,
            sample_index=None,
            snapshot_timestamp=None,
            fitting_plot_file=None,
            populations_plot_file=None,
        ):
            snapshot_path = self.output_dir / f"acceptance_snapshot_{snapshot_timestamp}.json"
            snapshot_path.write_text(
                (
                    f"sample={sample_index}\n"
                    f"fitting_plot={Path(fitting_plot_file).name}\n"
                    f"populations_plot={Path(populations_plot_file).name}\n"
                ),
                encoding="utf-8",
            )
            return snapshot_path

        def save_input_data_plot(self):
            return None

    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(cli_module, "StochasticSearch", FakeSearch)
    monkeypatch.setattr(cli_module, "build_timestamp_string", lambda: "20260502T120000000000")
    monkeypatch.setattr(cli_module.plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module.plt, "pause", lambda *args, **kwargs: None)

    exit_code = run_cli(
        [
            "search",
            "--data-dir",
            str(tmp_path / "data"),
            "--runtime-dir",
            str(runtime_dir),
            "--samples",
            "2",
        ]
    )

    captured = capsys.readouterr()
    fit_file = runtime_dir / "plots" / "fitting_DF_DHF_20260502T120000000000.png"
    accepted_log = runtime_dir / "accepted_samples.txt"
    acceptance_snapshot = runtime_dir / "parameters" / "acceptance_snapshot_20260502T120000000000.json"
    assert exit_code == 0
    assert "accepted_sample=1" in captured.out
    assert "acceptance_snapshot=" in captured.out
    assert fit_file.exists()
    assert acceptance_snapshot.exists()
    assert len(set(FakeSearch.figure_ids)) == 1
    assert accepted_log.read_text(encoding="utf-8").strip() == (
        "sample=1 snapshot=acceptance_snapshot_20260502T120000000000.json "
        "fitting_plot=fitting_DF_DHF_20260502T120000000000.png"
    )
