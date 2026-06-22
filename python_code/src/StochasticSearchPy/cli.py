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
    from .interactive_plot import launch_interactive_plot_app
    from .stochastic_search import StochasticSearch
except ImportError:
    from data_assets import copy_reference_dataset
    from data_processing import DataProcessing
    from interactive_plot import launch_interactive_plot_app
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
        help="Generate daily and weekly DF/DHF frequency tables from incidence CSVs.",
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
    search.add_argument(
        "--run-all",
        action="store_true",
        help="Evaluate every requested sample and save only the best sample at the end.",
    )
    search.set_defaults(func=run_search_command)

    interactive_plot = subparsers.add_parser(
        "interactive-plot",
        help="Launch a Streamlit frontend for interactive time-series plotting.",
    )
    interactive_plot.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV file to pre-load in the frontend.",
    )
    interactive_plot.add_argument(
        "--index-column",
        type=str,
        default=None,
        help="Optional timestamp column name. If omitted, the app uses the dataframe index.",
    )
    interactive_plot.add_argument("--host", type=str, default="127.0.0.1")
    interactive_plot.add_argument("--port", type=int, default=8501)
    interactive_plot.set_defaults(func=run_interactive_plot_command)

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
    """Generate daily and weekly DF/DHF frequency tables from incidence CSV files."""
    with DataProcessing(data_dir=args.data_dir) as processor:
        weekly_df, weekly_dhf = processor.build_weekly_frequency_tables()
        daily_df = processor.daily_df_frequency_table
        daily_dhf = processor.daily_dhf_frequency_table
        data_dir = processor.data_dir
    generated_files = (
        data_dir / "frequency_per_date_DF.csv",
        data_dir / "frequency_per_date_DHF.csv",
        data_dir / "frequency_per_week_DF.csv",
        data_dir / "frequency_per_week_DHF.csv",
    )
    print("generated:")
    for file_path in generated_files:
        print(file_path)
    print(
        "daily_rows_df={} daily_rows_dhf={} weekly_rows_df={} weekly_rows_dhf={}".format(
            len(daily_df),
            len(daily_dhf),
            len(weekly_df),
            len(weekly_dhf),
        )
    )
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


def run_interactive_plot_command(args) -> int:
    """Launch the Streamlit-based interactive plotting frontend."""
    return launch_interactive_plot_app(
        csv_path=args.csv,
        index_column=args.index_column,
        host=args.host,
        port=args.port,
    )


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


def rank_search_sample(sim, accepted: bool) -> tuple[float, float, float, float, float]:
    """Return a sortable rank where lower values identify a better sample."""
    machine_eps = np.finfo(np.float64).eps
    df_threshold = max(float(sim.df_error_threshold), machine_eps)
    dhf_threshold = max(float(sim.dhf_error_threshold), machine_eps)
    r_zero_threshold = float(getattr(sim, "R_ZERO_ACCEPTANCE_THRESHOLD", 1.0))
    peak_threshold = max(float(getattr(sim, "PEAK_DF_CASES_THRESHOLD", np.inf)), machine_eps)
    normalized_error = max(
        float(sim.df_fit_error) / df_threshold,
        float(sim.dhf_fit_error) / dhf_threshold,
    )
    r_zero_shortfall = max(0.0, r_zero_threshold - float(getattr(sim, "r_zero", 0.0)))
    peak_excess = max(0.0, float(sim.peak_df_cases) - peak_threshold) / peak_threshold
    return (
        0.0 if accepted else 1.0,
        normalized_error,
        r_zero_shortfall,
        peak_excess,
        float(sim.df_fit_error) + float(sim.dhf_fit_error),
    )


def capture_search_state(sim, sample_index: int, accepted: bool, rank: tuple) -> dict:
    """Capture enough model state to restore and save a selected sample later."""
    scalar_types = (int, float, bool, np.integer, np.floating, np.bool_)
    scalar_state = {}
    for key, value in sim.__dict__.items():
        if isinstance(value, scalar_types):
            scalar_state[key] = value.item() if hasattr(value, "item") else value

    state = {
        "sample_index": int(sample_index),
        "accepted": bool(accepted),
        "rank": tuple(rank),
        "scalars": scalar_state,
    }
    solution = getattr(sim, "solution", None)
    if solution is not None and hasattr(solution, "copy"):
        state["solution"] = solution.copy(deep=True)
    t_values = getattr(sim, "t", None)
    if t_values is not None:
        state["t"] = np.asarray(t_values, dtype=np.float64).copy()
    return state


def restore_search_state(sim, state: dict) -> None:
    """Restore a captured model state before writing best-sample artifacts."""
    for key, value in state.get("scalars", {}).items():
        setattr(sim, key, value)
    if "solution" in state:
        sim.solution = state["solution"].copy(deep=True)
        if hasattr(sim, "TIME_GRID_COLUMN") and sim.TIME_GRID_COLUMN in sim.solution:
            sim.t = sim.solution[sim.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
            sim.grid_size = len(sim.solution.index)
            if sim.grid_size > 1:
                sim.h = np.float64(sim.t[1] - sim.t[0])
    elif "t" in state:
        sim.t = state["t"].copy()


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
    acceptance_snapshot_file = None
    run_all = bool(args.run_all)
    best_state = None
    best_rank = None
    accepted_log = sim.runtime_dir / "accepted_samples.txt"
    best_log = sim.runtime_dir / "best_sample.txt"
    accepted_header = (
        "i       R_01        R_02        R_zero      error_DF    error_DHF   peak_DF     \n"
        "================================================================================"
    )
    fig = sim.create_reference_parameter_fitting_plot()
    fig.savefig(sim.plots_dir / "reference_parameter_fitting.png")
    plt.show()
    for i in np.arange(sim.sample_count):
        render_search_progress(int(i), sim.sample_count)
        sim.sample_model_parameters(flag_deterministic=False)
        sim.solve_ode_system()
        sim.compute_fitting_errors()
        if not run_all:
            live_fitting_figure = update_fitting_progress_figure(
                sim,
                fit_file=fit_file,
                previous_figure=live_fitting_figure,
            )
        error_df = sim.df_fit_error
        error_dhf = sim.dhf_fit_error
        r01_per_week, r02_per_week, r0_per_week = sim.compute_basic_reproduction_numbers()
        sim.r_zero = r0_per_week
        stop = sim.evaluate_search_acceptance()
        sample_rank = rank_search_sample(sim, accepted=stop)

        samples["i"].append(i)
        samples["r01"].append(r01_per_week)
        samples["r02"].append(r02_per_week)
        samples["r0"].append(r0_per_week)
        samples["err_df"].append(error_df)
        samples["err_dhf"].append(error_dhf)
        samples["peak_df_cases"].append(sim.peak_df_cases)
        samples["accepted"].append(stop)
        render_search_progress(int(i) + 1, sim.sample_count)

        if best_rank is None or sample_rank < best_rank:
            best_rank = sample_rank
            best_state = capture_search_state(
                sim,
                sample_index=int(i),
                accepted=stop,
                rank=sample_rank,
            )

        if stop and accepted_index is None:
            accepted_index = i

        if stop and not run_all:
            sim.save_solution_plots()
            acceptance_timestamp = build_timestamp_string()
            pop_file = sim.plots_dir / f"populations_grid_{acceptance_timestamp}.png"
            shutil.copy2(sim.plots_dir / "populations_grid.png", pop_file)
            acceptance_snapshot_file = sim.save_acceptance_snapshot(
                sample_index=i,
                snapshot_timestamp=acceptance_timestamp,
                fitting_plot_file=fit_file,
                populations_plot_file=pop_file,
            )
            with accepted_log.open("a", encoding="utf-8") as logf:
                logf.write(
                    f"sample={i} snapshot={acceptance_snapshot_file.name} "
                    f"fitting_plot={fit_file.name}\n"
                )
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
            print(f"acceptance_snapshot={acceptance_snapshot_file}")
            break
    else:
        sys.stdout.write("\n")

    if run_all and best_state is not None:
        restore_search_state(sim, best_state)
        best_index = int(best_state["sample_index"])
        best_accepted = bool(best_state["accepted"])
        sim.save_solution_plots()
        acceptance_timestamp = build_timestamp_string()
        best_fit_file = sim.plots_dir / f"fitting_DF_DHF_best_{acceptance_timestamp}.png"
        best_figure = sim.create_fitting_plot()
        best_figure.savefig(sim.plots_dir / "fitting_DF_DHF.png")
        best_figure.savefig(best_fit_file)
        plt.close(best_figure)
        pop_file = sim.plots_dir / f"populations_grid_{acceptance_timestamp}.png"
        shutil.copy2(sim.plots_dir / "populations_grid.png", pop_file)
        acceptance_snapshot_file = sim.save_acceptance_snapshot(
            sample_index=best_index,
            snapshot_timestamp=acceptance_timestamp,
            fitting_plot_file=best_fit_file,
            populations_plot_file=pop_file,
        )
        with best_log.open("w", encoding="utf-8") as logf:
            logf.write(
                f"sample={best_index} accepted={best_accepted} "
                f"snapshot={acceptance_snapshot_file.name} "
                f"fitting_plot={best_fit_file.name}\n"
            )
        print(accepted_header)
        print(
            "%-8d%-12f%-12f%-12f%-12f%-12f%-12f"
            % (
                best_index,
                np.sqrt(float(getattr(sim, "r_01", 0.0))),
                np.sqrt(float(getattr(sim, "r_02", 0.0))),
                float(getattr(sim, "r_zero", 0.0)),
                sim.df_fit_error,
                sim.dhf_fit_error,
                sim.peak_df_cases,
            )
        )
        print(f"best_sample={best_index}")
        print(f"best_sample_accepted={best_accepted}")
        print(f"best_score={best_state['rank']}")
        print(f"acceptance_snapshot={acceptance_snapshot_file}")

    plt.figure(figsize=(7, 5))
    plt.scatter(samples["err_df"], samples["r0"], s=10, alpha=0.4, label="Samples")
    highlighted_index = best_state["sample_index"] if run_all and best_state is not None else accepted_index
    if highlighted_index is not None:
        idx = samples["i"].index(highlighted_index)
        plt.scatter(
            [samples["err_df"][idx]],
            [samples["r0"][idx]],
            color="red",
            s=60,
            label="Best" if run_all else "Accepted",
        )
    plt.xlabel("Error DF")
    plt.ylabel("R0")
    plt.title("Search progress across all samples" if run_all else "Search progress until acceptance")
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
