"""Command-line interface for the Two Strains Dengue HMS package."""

import argparse
import datetime
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .data_assets import prepare_data_directory
    from .data_processing import DataProcessing
    from .stochastic_search import StochasticSearch
except ImportError:
    from data_assets import prepare_data_directory
    from data_processing import DataProcessing
    from stochastic_search import StochasticSearch


def _timestamp() -> str:
    """Return a filesystem-safe timestamp used in generated artifact names."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def build_parser() -> argparse.ArgumentParser:
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
    smoke.set_defaults(func=run_smoke)

    tables = subparsers.add_parser(
        "frequency-tables",
        help="Generate weekly DF and DHF frequency tables from incidence CSVs.",
    )
    tables.add_argument("--data-dir", type=Path, default=None)
    tables.set_defaults(func=run_frequency_tables)

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
    prepare_data.set_defaults(func=run_prepare_data)

    search = subparsers.add_parser(
        "search",
        help="Run the stochastic search and persist artifacts under the runtime directory.",
    )
    search.add_argument("--data-dir", type=Path, default=None)
    search.add_argument("--runtime-dir", type=Path, default=None)
    search.add_argument("--samples", type=int, default=1)
    search.add_argument("--bound-error-fd", type=float, default=100.0)
    search.add_argument("--bound-error-fhd", type=float, default=25.0)
    search.set_defaults(func=run_search)

    return parser


def run_smoke(args) -> int:
    """Run a lightweight model instantiation check and print R0 metrics."""
    sim = StochasticSearch(data_dir=args.data_dir, runtime_dir=args.runtime_dir)
    r01_per_week, r02_per_week, r0_per_week = sim.compute_r_zero()
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


def run_frequency_tables(args) -> int:
    """Generate weekly DF and DHF frequency tables from incidence CSV files."""
    with DataProcessing(data_dir=args.data_dir) as processor:
        freq_df, freq_dhf = processor.incidence_frequency_tables()
        data_dir = processor.data_dir
    print(
        "generated={} and {}".format(
            data_dir / "frecuency_per_week_DF.csv",
            data_dir / "frecuency_per_week_DHF.csv",
        )
    )
    print(f"df_rows={len(freq_df)} dhf_rows={len(freq_dhf)}")
    return 0


def run_prepare_data(args) -> int:
    """Copy the committed reference dataset into a working data directory."""
    source_dir, output_dir, copied_files = prepare_data_directory(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
    )
    print(f"source_dir={source_dir}")
    print(f"output_dir={output_dir}")
    print(f"copied_files={len(copied_files)}")
    return 0


def run_search(args) -> int:
    """Execute the stochastic search workflow and persist generated artifacts."""
    sim = StochasticSearch(data_dir=args.data_dir, runtime_dir=args.runtime_dir)
    sim.number_of_samples = args.samples
    sim.bound_error_FD = args.bound_error_fd
    sim.bound_error_FHD = args.bound_error_fhd
    sim.incidence_frequency_tables()

    print(
        "%-8s%-12s%-12s%-12s%-12s%-12s%-12s"
        % ("i", "R_01", "R_02", "R_zero", "error_DF", "error_DHF", "z_max")
    )

    samples = {
        "i": [],
        "r01": [],
        "r02": [],
        "r0": [],
        "err_df": [],
        "err_dhf": [],
        "zmax": [],
        "accepted": [],
    }

    accepted_index = None
    accepted_log = sim.runtime_dir / "accepted_samples.txt"

    for i in np.arange(sim.number_of_samples):
        sim.parameters_sampling(flag_deterministic=False)
        sim.ode_int_solution()
        sim.fitting_error()
        error_df = sim.fitting_error_DF
        error_dhf = sim.fitting_error_DHF
        r01_per_week, r02_per_week, r0_per_week = sim.compute_r_zero()
        stop = sim.update_conditions_search()

        samples["i"].append(i)
        samples["r01"].append(r01_per_week)
        samples["r02"].append(r02_per_week)
        samples["r0"].append(r0_per_week)
        samples["err_df"].append(error_df)
        samples["err_dhf"].append(error_dhf)
        samples["zmax"].append(sim.z_max)
        samples["accepted"].append(stop)

        print(
            "%-12d%-12f%-12f%-12f%-12f%-12f%-12f"
            % (
                i,
                r01_per_week,
                r02_per_week,
                r0_per_week,
                error_df,
                error_dhf,
                sim.z_max,
            )
        )

        if stop and accepted_index is None:
            accepted_index = i
            sim.solution_plot()
            sim.fitting_plot()
            sim.save_parameters()
            timestamp = _timestamp()
            fit_file = sim.plots_dir / f"fitting_DF_DHF_{timestamp}.png"
            pop_file = sim.plots_dir / f"populations_grid_{timestamp}.png"
            shutil.copy2(sim.plots_dir / "fitting_DF_DHF.png", fit_file)
            shutil.copy2(sim.plots_dir / "populations_grid.png", pop_file)
            with accepted_log.open("a", encoding="utf-8") as logf:
                logf.write(f"{fit_file.name}\n")
            print(f"accepted_sample={i}")
            break

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
    sim.plot_input_data()
    return 0


def main(argv=None) -> int:
    """Run the CLI and return the selected subcommand exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
