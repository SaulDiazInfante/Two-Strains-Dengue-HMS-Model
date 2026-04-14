try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

from StochasticSearchPy import data_assets
from StochasticSearchPy.cli import run_cli


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
