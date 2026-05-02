"""Command-line interface for the Two Strains Dengue HMS package."""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .data_assets import copy_reference_dataset
    from .data_processing import DataProcessing
    from .stochastic_search import StochasticSearch
except ImportError:
    from data_assets import copy_reference_dataset
    from data_processing import DataProcessing
    from stochastic_search import StochasticSearch


def build_timestamp_string() -> str:
    """Return a filesystem-safe timestamp used in generated artifact names."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")


def update_fitting_progress_figure(sim, fit_file: Path, previous_figure=None):
    """Save and refresh the current DF/DHF fitting figure for an in-progress search."""
    figure = sim.create_fitting_plot(figure=previous_figure)
    figure.savefig(sim.plots_dir / "fitting_DF_DHF.png")
    figure.savefig(fit_file)

    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(fit_file.name)

    if previous_figure is None:
        plt.show(block=False)

    canvas = getattr(figure, "canvas", None)
    if canvas is not None and hasattr(canvas, "draw_idle"):
        canvas.draw_idle()
    if canvas is not None and hasattr(canvas, "flush_events"):
        canvas.flush_events()
    plt.pause(0.001)
    return figure


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="two-strains-dengue",
        description="Run the Two Strains Dengue HMS model utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Instantiate the model and print a basic reproduction summary.",
    )
    smoke.add_argument("--data-dir", type=Path, default=None)
    smoke.add_argument("--runtime-dir", type=Path, default=None)
    smoke.set_defaults(func=run_smoke_command)

    tables = subparsers.add_parser(
        "frequency-tables",
        help="Generate weekly DF and DHF frequency tables from incidence CSVs.",
    )
    tables.add_argument("--data-dir", type=Path, default=None)
    tables.set_defaults(func=run_frequency_tables_command)

    prepare_data = subparsers.add_parser(
        "prepare-data",
        help="Copy the reference dataset into a working directory for local runs.",
    )
    prepare_data.add_argument("--source-dir", type=Path, default=None)
    prepare_data.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
    )
    prepare_data.set_defaults(func=run_prepare_data_command)

    search = subparsers.add_parser(
        "search",
        help="Run the stochastic search and persist artifacts under the runtime directory.",
    )
    search.add_argument("--data-dir", type=Path, default=None)
    search.add_argument("--runtime-dir", type=Path, default=None)
    search.add_argument("--samples", type=int, default=1)
    search.add_argument("--bound-error-fd", type=float, default=300.0)
    search.add_argument("--bound-error-fhd", type=float, default=25.0)
    search.set_defaults(func=run_search_command)

    return parser


def run_smoke_command(args) -> int:
    """Run a lightweight model instantiation check and print R0 metrics."""
    sim = StochasticSearch(data_dir=args.data_dir, runtime_dir=args.runtime_dir)
    r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()
    print(f"data_dir={sim.data_dir}")
    print(f"runtime_dir={sim.runtime_dir}")
    print(
        "r01={:.6f} r02={:.6f} r0={:.6f}".format(
            r01_per_week,
            r02_per_week,
            r0_per_week,
        )
    )
    return 0


def run_frequency_tables_command(args) -> int:
    """Generate weekly DF and DHF frequency tables from incidence CSV files."""
    with DataProcessing(data_dir=args.data_dir) as processor:
        freq_df, freq_dhf = processor.build_weekly_frequency_tables()
        data_dir = processor.data_dir
    print(
        "generated={} and {}".format(
            data_dir / "frequency_per_week_DF.csv",
            data_dir / "frequency_per_week_DHF.csv",
        )
    )
    print(f"df_rows={len(freq_df)} dhf_rows={len(freq_dhf)}")
    return 0


def run_prepare_data_command(args) -> int:
    """Copy the committed reference dataset into a working data directory."""
    source_dir, output_dir, copied_files = copy_reference_dataset(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
    )
    print(f"source_dir={source_dir}")
    print(f"output_dir={output_dir}")
    print(f"copied_files={len(copied_files)}")
    return 0

def render_search_progress(sample_number: int, sample_count: int, width: int = 40) -> None:
    """Render the in-place stochastic search progress bar."""
    if sample_count <= 0:
        return

    completed = min(sample_number, sample_count)
    fraction_completed = completed / sample_count
    filled_width = int(width * fraction_completed)
    bar = "#" * filled_width + "-" * (width - filled_width)
    sys.stdout.write(
        "\rsearch progress: |{}| {:6.2f}% ({}/{})".format(
            bar,
            100.0 * fraction_completed,
            completed,
            sample_count,
        )
    )
    sys.stdout.flush()


def run_search_command(args) -> int:
    """Execute the stochastic search workflow and persist generated artifacts."""
    sim = StochasticSearch(data_dir=args.data_dir, runtime_dir=args.runtime_dir)
    sim.sample_count = args.samples
    sim.df_error_threshold = args.bound_error_fd
    sim.dhf_error_threshold = args.bound_error_fhd
    sim.build_weekly_frequency_tables()
    search_timestamp = build_timestamp_string()
    fit_file = sim.plots_dir / f"fitting_DF_DHF_{search_timestamp}.png"
    live_fitting_figure = None
    samples = {
        "i": [],
        "r01": [],
        "r02": [],
        "r0": [],
        "err_df": [],
        "err_dhf": [],
        "peak_df_cases": [],
        "accepted": [],
    }

    accepted_index = None
    accepted_log = sim.runtime_dir / "accepted_samples.txt"
    accepted_header = (
        "i       R_01        R_02        R_zero      error_DF    error_DHF   peak_DF     \n"
        "================================================================================"
    )

    for i in np.arange(sim.sample_count):
        render_search_progress(int(i), sim.sample_count)
        sim.sample_model_parameters(flag_deterministic=False)
        sim.solve_ode_system()
        sim.compute_fitting_errors()
        live_fitting_figure = update_fitting_progress_figure(
            sim,
            fit_file=fit_file,
            previous_figure=live_fitting_figure,
        )
        error_df = sim.df_fit_error
        error_dhf = sim.dhf_fit_error
        r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()
        stop = sim.evaluate_search_acceptance()

        samples["i"].append(i)
        samples["r01"].append(r01_per_week)
        samples["r02"].append(r02_per_week)
        samples["r0"].append(r0_per_week)
        samples["err_df"].append(error_df)
        samples["err_dhf"].append(error_dhf)
        samples["peak_df_cases"].append(sim.peak_df_cases)
        samples["accepted"].append(stop)
        render_search_progress(int(i) + 1, sim.sample_count)

        if stop and accepted_index is None:
            accepted_index = i
            sim.save_solution_plots()
            sim.save_parameter_snapshot()
            timestamp = build_timestamp_string()
            pop_file = sim.plots_dir / f"populations_grid_{timestamp}.png"
            shutil.copy2(sim.plots_dir / "populations_grid.png", pop_file)
            with accepted_log.open("a", encoding="utf-8") as logf:
                logf.write(f"{fit_file.name}\n")
            sys.stdout.write("\n")
            print(accepted_header)
            print(
                "%-8d%-12f%-12f%-12f%-12f%-12f%-12f"
                % (
                    i,
                    r01_per_week,
                    r02_per_week,
                    r0_per_week,
                    error_df,
                    error_dhf,
                    sim.peak_df_cases,
                )
            )
            print(f"accepted_sample={i}")
            break
    else:
        sys.stdout.write("\n")

    plt.figure(figsize=(7, 5))
    plt.scatter(samples["err_df"], samples["r0"], s=10, alpha=0.4, label="Samples")
    if accepted_index is not None:
        idx = samples["i"].index(accepted_index)
        plt.scatter(
            [samples["err_df"][idx]],
            [samples["r0"][idx]],
            color="red",
            s=60,
            label="Accepted",
        )
    plt.xlabel("Error DF")
    plt.ylabel("R0")
    plt.title("Accepted samples search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sim.plots_dir / "accepted_samples.png")
    plt.close()
    sim.save_input_data_plot()
    return 0


def run_cli(argv=None) -> int:
    """Run the CLI and return the selected subcommand exit code."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(run_cli())
