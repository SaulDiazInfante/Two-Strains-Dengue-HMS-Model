"""Core stochastic search model and plotting utilities."""

import json
import numpy as np
import pandas as pd
import datetime
import yaml
from scipy import integrate
import matplotlib.pyplot as plt
import operator
from pathlib import Path


# Local import used lazily inside __init__ if raw_data files are missing.
try:
    from . import data_processing
    from .data_assets import get_repository_root
except ImportError:
    # Fallback for running as a script (python stochastic_searchPy.py)
    import data_processing
    from data_assets import get_repository_root


class StochasticSearch(data_processing.DataProcessing):
    """
    This class implements a stochastic search algorithm for epidemiological modeling.
    Its purpose is to process weekly frequency data, configure and manage model parameters,
    evaluate search acceptance criteria, and generate fitting plots.

    :ivar sample_count: Number of samples used in the stochastic search.
    :type sample_count: int
    :ivar base_dir: Base directory of the current script.
    :type base_dir: pathlib.Path
    :ivar runtime_dir: Directory for storing runtime artifacts.
    :type runtime_dir: pathlib.Path
    :ivar plots_dir: Directory for saving plot figures.
    :type plots_dir: pathlib.Path
    :ivar output_dir: Directory for saving parameter files.
    :type output_dir: pathlib.Path
    :ivar weekly_df_frequency_array: Processed frequency array for DF (Dengue Fever) data.
    :type weekly_df_frequency_array: numpy.ndarray
    :ivar weekly_dhf_frequency_array: Processed frequency array for DHF (Dengue Hemorrhagic Fever) data.
    :type weekly_dhf_frequency_array: numpy.ndarray
    :ivar df_error_threshold: Error threshold for DF fitting.
    :type df_error_threshold: float
    :ivar dhf_error_threshold: Error threshold for DHF fitting.
    :type dhf_error_threshold: float
    :ivar Lambda_M: Initial mosquito carrying capacity.
    :type Lambda_M: float
    :ivar beta_M: Mosquito-human biting rate factor.
    :type beta_M: float
    :ivar beta_H: Human-mosquito biting rate factor.
    :type beta_H: float
    :ivar b: Specific growth parameter for the mosquito population.
    :type b: float
    :ivar mu_M: Mosquito death rate.
    :type mu_M: float
    :ivar mu_H: Human death rate.
    :type mu_H: float
    :ivar alpha_c: Complex transmission rate.
    :type alpha_c: float
    :ivar alpha_h: Hemorrhagic transmission rate.
    :type alpha_h: float
    :ivar sigma: Mosquito-to-human infectious transmission parameter.
    :type sigma: float
    :ivar p: Fraction of surviving mosquitoes.
    :type p: float
    :ivar theta: Specificity parameter for the dynamics model.
    :type theta: float
    :ivar peak_df_cases: Peak case count for DF in simulation.
    :type peak_df_cases: float
    :ivar M_s0: Initial adult mosquito population size.
    :type M_s0: float
    :ivar M_10: Initial mosquito infection state 1.
    :type M_10: float
    :ivar M_20: Initial mosquito infection state 2.
    :type M_20: float
    :ivar I_10: Initial human infective state 1 population.
    :type I_10: float
    :ivar I_20: Initial human infective state 2 population.
    :type I_20: float
    :ivar S_0: Initial susceptible human population.
    :type S_0: float
    :ivar S_m1_0: Initial pre-symptomatic mosquito population.
    :type S_m1_0: float
    :ivar Y_m1_c0: Initial mosquito compartments for particular symptoms.
    :type Y_m1_c0: float
    :ivar Y_m1_h0: Initial hemorrhagic state mosquito compartments.
    :type Y_m1_h0: float
    :ivar Rec_0: Initial recovered human population.
    :type Rec_0: float
    :ivar z0: Initial average mosquito infectivity parameter.
    :type z0: float
    :ivar N_S: Total human population size.
    :type N_S: float
    :ivar N_S_m1: Total susceptible and pre-symptomatic mosquito population size.
    :type N_S_m1: float
    :ivar Lambda_S_m1: Initial birth rate for resting mosquitoes.
    :type Lambda_S_m1: float
    :ivar Lambda_S: Initial birth rate for surviving mosquitoes.
    :type Lambda_S: float
    :ivar t0: Start time of simulation.
    :type t0: int
    :ivar T: End time of simulation.
    :type T: int
    :ivar grid_size: Computation grid points for simulation.
    :type grid_size: int
    :ivar h: Time step for computation based on grid size.
    :type h: numpy.float64
    :ivar r_01: Auxiliary performance metric for the model.
    :type r_01: float
    :ivar r_02: Another auxiliary performance metric for the model.
    :type r_02: float
    :ivar r_zero: Basic reproduction number for initial condition.
    :type r_zero: int
    :ivar t: Time grid for simulation.
    :type t: numpy.ndarray
    :ivar solution: ODE solution table storing the time grid and compartment trajectories.
    :type solution: pandas.DataFrame
    :ivar df_fit_error: Error metric for DF fitting.
    :type df_fit_error: float
    :ivar dhf_fit_error: Error metric for DHF fitting.
    :type dhf_fit_error: float
    :ivar meets_r_zero_threshold: Boolean indicator for R0 threshold evaluation.
    :type meets_r_zero_threshold: bool
    :ivar meets_df_error_threshold: Boolean indicator for DF error threshold.
    :type meets_df_error_threshold: bool
    :ivar meets_dhf_error_threshold: Boolean indicator for DHF error threshold.
    :type meets_dhf_error_threshold: bool
    :ivar meets_acceptance_criteria: Boolean flag for overall acceptance criteria.
    :type meets_acceptance_criteria: bool
    """

    SAMPLED_PARAMETER_COLUMNS = (
        "Lambda_M",
        "beta_M",
        "beta_H",
        "b",
        "mu_M",
        "alpha_c",
        "alpha_h",
        "sigma",
        "p",
        "theta",
        "M_s0",
        "M_10",
        "M_20",
        "S_0",
        "I_10",
        "I_20",
        "S_m1_0",
        "Y_m1_c0",
        "Y_m1_h0",
        "R_s0",
        "R_s_m1_0",
        "z0",
        "h",
        "T",
    )
    ODE_ARGUMENT_NAMES = (
        "Lambda_M",
        "Lambda_S",
        "Lambda_S_m1",
        "beta_M",
        "beta_H",
        "b",
        "mu_M",
        "mu_H",
        "alpha_c",
        "alpha_h",
        "sigma",
        "p",
        "theta",
    )
    ODE_STATE_COLUMNS = (
        "M_s",
        "M_I1",
        "M_I2",
        "S",
        "I_1",
        "I_2",
        "R_s",
        "S_m1",
        "Y_m1_c",
        "Y_m1_h",
        "R_s_m1",
        "z",
    )
    TIME_GRID_COLUMN = "time_grid"
    TIME_DAY_COLUMN = "time_day"
    ODE_SOLUTION_COLUMNS = (TIME_GRID_COLUMN, *ODE_STATE_COLUMNS)
    FITTING_PLOT_FIGSIZE = (18.0, 8.0)
    FITTING_PLOT_LAYOUT = {
        "left": 0.055,
        "right": 0.99,
        "bottom": 0.09,
        "top": 0.92,
        "wspace": 0.22,
        "hspace": 0.34,
    }
    DAYS_PER_WEEK = 7
    DF_FITTING_START_INDEX = 10
    DHF_FITTING_START_INDEX = 6
    FITTING_ERROR_START_WEEK = 37.0
    FITTING_ERROR_END_WEEK = 47.0
    PLOT_END_WEEK = 53.0
    FITTING_WINDOW_WEEKS = int(FITTING_ERROR_END_WEEK - FITTING_ERROR_START_WEEK)
    MOVING_AVERAGE_WINDOW_DAYS = DAYS_PER_WEEK
    # The plot shows the full model-aligned trajectory, while fit metrics are
    # computed only on the requested sub-interval.
    R_ZERO_ACCEPTANCE_THRESHOLD = 1.0
    PEAK_DF_CASES_THRESHOLD = 700.0
    DEFAULT_MODEL_PARAMETER_FILE = Path("data/default_model_parameters.json")
    PARAMETER_GROUP_KEYS = (
        "recruitment_parameters",
        "transmission_parameters",
        "vital_dynamics_parameters",
        "clinical_progression_parameters",
        "risk_and_reporting_parameters",
        "initial_conditions",
        "derived_population_parameters",
        "integration_parameters",
        "state_tracking_parameters",
    )

    def __init__(self, data_dir=None, runtime_dir=None):
        super().__init__(data_dir=data_dir)
        self.sample_count = 30
        self.base_dir = Path(__file__).resolve().parent
        self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else Path.cwd() / "artifacts"
        df_path = Path(self.build_data_file_path("frequency_per_week_DF.csv"))
        dhf_path = Path(self.build_data_file_path("frequency_per_week_DHF.csv"))
        self.plots_dir = self.runtime_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.runtime_dir / "parameters"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # If the weekly CSVs are missing, build them from the source incidence data.
        if not df_path.exists() or not dhf_path.exists():
            self.build_weekly_frequency_tables()
        self.weekly_df_frequency_array = self.read_weekly_frequency_array(df_path)
        self.weekly_dhf_frequency_array = self.read_weekly_frequency_array(dhf_path)
        self.df_error_threshold = 300
        self.dhf_error_threshold = 150
        default_parameter_path = get_repository_root() / self.DEFAULT_MODEL_PARAMETER_FILE
        self.load_parameters_from_json(default_parameter_path, strict=True)
        #
        self.df_fit_error = 0.0
        self.dhf_fit_error = 0.0
        self.df_weekly_fit_error = 0.0
        self.dhf_weekly_fit_error = 0.0
        self.meets_r_zero_threshold = False
        self.meets_df_error_threshold = False
        self.meets_dhf_error_threshold = False
        self.meets_acceptance_criteria = False

    @staticmethod
    def read_weekly_frequency_array(path: Path) -> np.ndarray:
        """Load a weekly frequency CSV, accepting integer or date-like week labels."""
        frame = pd.read_csv(path)
        week = pd.to_numeric(frame["week"], errors="coerce")
        if week.isna().any():
            week = pd.to_datetime(frame["week"]).dt.isocalendar().week.astype(int)
        counts = pd.to_numeric(frame["count"]).astype(int)
        return np.column_stack([week.astype(int).to_numpy(), counts.to_numpy()])

    def evaluate_search_acceptance(self):
        """Evaluate whether the current sample satisfies acceptance criteria."""
        meets_r_zero_threshold = self.r_zero > self.R_ZERO_ACCEPTANCE_THRESHOLD
        meets_df_error_threshold = self.df_fit_error < self.df_error_threshold
        meets_dhf_error_threshold = self.dhf_fit_error < self.dhf_error_threshold
        meets_peak_df_threshold = self.peak_df_cases < self.PEAK_DF_CASES_THRESHOLD
        meets_acceptance_criteria = (
            meets_df_error_threshold
            and meets_dhf_error_threshold
            and meets_r_zero_threshold
            and meets_peak_df_threshold
        )

        self.meets_acceptance_criteria = meets_acceptance_criteria
        self.meets_r_zero_threshold = meets_r_zero_threshold
        self.meets_df_error_threshold = meets_df_error_threshold
        self.meets_dhf_error_threshold = meets_dhf_error_threshold
        return meets_acceptance_criteria

    @staticmethod
    def _sample_weekly_solution_points(
        time_grid: np.ndarray,
        solution_values: np.ndarray,
        sample_stride: int,
        trim_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a solution series at the weekly locations used during fitting."""
        sampled_weeks = np.round(np.asarray(time_grid)[:-1:sample_stride]).astype(int)
        sampled_values = np.asarray(solution_values)[:-1:sample_stride]
        return (
            np.delete(sampled_weeks, trim_indices),
            np.delete(sampled_values, trim_indices),
        )

    def _load_daily_frequency_tables_for_fitting(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load daily DF/DHF tables, rebuilding them in memory if persisted CSVs are invalid."""
        if (
            not self.daily_df_frequency_table.empty
            and not self.daily_dhf_frequency_table.empty
        ):
            return self.daily_df_frequency_table, self.daily_dhf_frequency_table

        df_path = Path(self.build_data_file_path("frequency_per_date_DF.csv"))
        dhf_path = Path(self.build_data_file_path("frequency_per_date_DHF.csv"))
        try:
            df_counts = self.read_daily_frequency_table(df_path)
            dhf_counts = self.read_daily_frequency_table(dhf_path)
        except (OSError, KeyError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            df_counts, dhf_counts = self.build_daily_frequency_tables(persist=False)

        self.daily_df_frequency_table = df_counts
        self.daily_dhf_frequency_table = dhf_counts
        return df_counts, dhf_counts

    @staticmethod
    def _get_iso_week_start_date(observed_counts: pd.DataFrame, time_week: float) -> pd.Timestamp:
        """Return the calendar day aligned with ``time_week`` in the data year."""
        if observed_counts.empty:
            raise ValueError("Observed daily counts are empty; cannot resolve a fitting window.")

        iso_year = int(observed_counts.index.min().isocalendar().year)
        base_week = int(np.floor(float(time_week)))
        base_week = max(base_week, 1)
        last_iso_week = int(datetime.date(iso_year, 12, 28).isocalendar().week)
        base_week = min(base_week, last_iso_week)
        week_fraction = float(time_week) - np.floor(float(time_week))
        day_offset = int(np.round(week_fraction * StochasticSearch.DAYS_PER_WEEK))
        return (
            pd.Timestamp(datetime.date.fromisocalendar(iso_year, base_week, 1))
            + pd.Timedelta(days=day_offset)
        )

    def _build_daily_observed_fitting_series(
        self,
        observed_counts: pd.DataFrame,
        start_time_week: float,
        end_time_week: float,
    ) -> pd.DataFrame:
        """Return a dense daily observed series over the requested interval."""
        total_days = int(
            round((float(end_time_week) - float(start_time_week)) * self.DAYS_PER_WEEK)
        )
        if total_days <= 0:
            raise ValueError(
                "end_time_week must be greater than start_time_week to build an observed series."
            )
        start_date = self._get_iso_week_start_date(observed_counts, start_time_week)
        day_index = pd.date_range(start=start_date, periods=total_days, freq="D", name="date")
        regularized_series = observed_counts.reindex(day_index, fill_value=0.0)
        return regularized_series

    @staticmethod
    def _build_daily_simulated_fitting_series(
        time_grid: np.ndarray,
        simulated_values: np.ndarray,
        start_time_week: float,
        day_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Interpolate a simulated weekly-time series onto one point per day."""
        daily_time_points = float(start_time_week) + (
            np.arange(len(day_index), dtype=np.float64) / StochasticSearch.DAYS_PER_WEEK
        )
        daily_values = np.interp(daily_time_points, time_grid, simulated_values)
        df_daily_values = pd.DataFrame(
            {"count": daily_values},
            index=day_index,
            dtype=np.float64,
        )
        return df_daily_values

    def _build_daily_fitting_frame(
        self,
        observed_counts: pd.DataFrame,
        time_grid: np.ndarray,
        simulated_values: np.ndarray,
        start_time_week: float,
        end_time_week: float,
    ) -> pd.DataFrame:
        """Build the observed/simulated daily fitting series for one outcome."""
        observed_daily = self._build_daily_observed_fitting_series(
            observed_counts,
            start_time_week,
            end_time_week,
        )
        simulated_daily = self._build_daily_simulated_fitting_series(
            time_grid,
            simulated_values,
            start_time_week,
            observed_daily.index,
        )
        observed_average = self.compute_moving_average(
            observed_daily,
            self.MOVING_AVERAGE_WINDOW_DAYS,
        )
        simulated_average = self.compute_moving_average(
            simulated_daily,
            self.MOVING_AVERAGE_WINDOW_DAYS,
        )
        fitting_frame = pd.DataFrame(
            {
                "time_week": float(start_time_week)
                + (np.arange(len(observed_daily), dtype=np.float64) / self.DAYS_PER_WEEK),
                "observed_daily": observed_daily["count"].to_numpy(dtype=np.float64),
                "simulated_daily": simulated_daily["count"].to_numpy(dtype=np.float64),
                "observed_moving_average": observed_average.to_numpy(dtype=np.float64),
                "simulated_moving_average": simulated_average.to_numpy(dtype=np.float64),
            },
            index=observed_daily.index,
        )
        fitting_frame.index.name = "date"
        return fitting_frame

    @staticmethod
    def _compute_moving_average_fit_error(
        fitting_frame: pd.DataFrame,
        observed_column: str = "observed_moving_average",
        simulated_column: str = "simulated_moving_average",
    ) -> float:
        """Compute an infinity-norm fit error from an observed/simulated comparison frame."""
        valid_rows = fitting_frame.dropna(
            subset=[observed_column, simulated_column]
        )
        return float(
            np.linalg.norm(
                valid_rows[observed_column].to_numpy()
                - valid_rows[simulated_column].to_numpy(),
                ord=np.inf,
            )
        )

    def _build_comparison_frames_for_interval(
        self,
        start_time_week: float,
        end_time_week: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return DF and DHF comparison frames for the requested interval."""
        solution_frame = self.solution
        time_grid = solution_frame[self.TIME_GRID_COLUMN].to_numpy()
        z = solution_frame["z"].to_numpy()

        Y_m1_h = solution_frame["Y_m1_h"].to_numpy()
        daily_df_counts, daily_dhf_counts = self._load_daily_frequency_tables_for_fitting()
        df_frame = self._build_daily_fitting_frame(
            daily_df_counts,
            time_grid,
            z,
            start_time_week,
            end_time_week,
        )
        dhf_frame = self._build_daily_fitting_frame(
            daily_dhf_counts,
            time_grid,
            Y_m1_h,
            start_time_week,
            end_time_week,
        )
        return df_frame, dhf_frame

    def _resolve_plot_time_bounds(self, time_grid: np.ndarray) -> tuple[float, float]:
        """Return the displayed time range for observation-aligned fitting panels."""
        start_time_week = self._resolve_fitting_start_time_week(time_grid)
        end_time_week = min(float(np.max(time_grid)), float(self.PLOT_END_WEEK))
        if end_time_week <= start_time_week:
            raise ValueError("Plot end week must be greater than the plot start week.")
        return start_time_week, end_time_week

    def _resolve_error_time_bounds(self, time_grid: np.ndarray) -> tuple[float, float]:
        """Return the time interval used for fit-error computation."""
        plot_start_week, plot_end_week = self._resolve_plot_time_bounds(time_grid)
        start_time_week = max(float(self.FITTING_ERROR_START_WEEK), plot_start_week)
        end_time_week = min(float(self.FITTING_ERROR_END_WEEK), plot_end_week)
        if end_time_week <= start_time_week:
            raise ValueError("Error end week must be greater than error start week.")
        return start_time_week, end_time_week

    def _build_plot_comparison_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return DF and DHF comparison frames for the displayed time range."""
        time_grid = self.solution[self.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
        start_time_week, end_time_week = self._resolve_plot_time_bounds(time_grid)
        return self._build_comparison_frames_for_interval(start_time_week, end_time_week)

    def _build_fitting_comparison_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the DF and DHF comparison frames used for fit-error computation."""
        time_grid = self.solution[self.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
        start_time_week, end_time_week = self._resolve_error_time_bounds(time_grid)
        return self._build_comparison_frames_for_interval(start_time_week, end_time_week)

    @staticmethod
    def _aggregate_numeric_series_by_week(frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a date-indexed numeric count series into week-start totals."""
        if frame.empty:
            return pd.DataFrame(
                {
                    "week": pd.Series(dtype="datetime64[ns]"),
                    "count": pd.Series(dtype=float),
                }
            )

        week_start = pd.DatetimeIndex(pd.to_datetime(frame.index).normalize()).to_period(
            "W-SUN"
        ).start_time
        weekly = (
            frame.reset_index(drop=False)
            .assign(week=week_start)
            .groupby("week", as_index=False)["count"]
            .sum()
            .sort_values("week")
            .reset_index(drop=True)
        )
        weekly["week"] = pd.to_datetime(weekly["week"]).dt.normalize()
        weekly["count"] = pd.to_numeric(weekly["count"], errors="raise").astype(np.float64)
        return weekly

    def _build_weekly_fitting_frame(self, daily_fitting_frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a daily fitting frame onto weekly observation-aligned totals."""
        if daily_fitting_frame.empty:
            return pd.DataFrame(
                {
                    "week": pd.Series(dtype="datetime64[ns]"),
                    "time_week": pd.Series(dtype=np.float64),
                    "observed_weekly": pd.Series(dtype=np.float64),
                    "simulated_weekly": pd.Series(dtype=np.float64),
                    "observed_weekly_moving_average": pd.Series(dtype=np.float64),
                    "simulated_weekly_moving_average": pd.Series(dtype=np.float64),
                }
            )

        observed_weekly = self._aggregate_numeric_series_by_week(
            daily_fitting_frame.loc[:, ["observed_daily"]].rename(
                columns={"observed_daily": "count"}
            )
        ).rename(columns={"count": "observed_weekly"})
        simulated_weekly = self._aggregate_numeric_series_by_week(
            daily_fitting_frame.loc[:, ["simulated_daily"]].rename(
                columns={"simulated_daily": "count"}
            )
        ).rename(columns={"count": "simulated_weekly"})

        daily_dates = pd.DatetimeIndex(pd.to_datetime(daily_fitting_frame.index).normalize())
        weekly_time_map = (
            daily_fitting_frame.reset_index(drop=False)
            .assign(week=daily_dates.to_period("W-SUN").start_time)
            .groupby("week", as_index=False)["time_week"]
            .min()
            .sort_values("week")
            .reset_index(drop=True)
        )
        weekly_time_map["week"] = pd.to_datetime(weekly_time_map["week"]).dt.normalize()

        weekly_frame = (
            observed_weekly.merge(simulated_weekly, on="week", how="outer")
            .merge(weekly_time_map, on="week", how="left")
            .sort_values("week")
            .reset_index(drop=True)
        )
        weekly_frame["observed_weekly_moving_average"] = self.compute_moving_average(
            weekly_frame["observed_weekly"],
            self.WEEKLY_MOVING_AVERAGE_WINDOW_WEEKS,
        ).to_numpy(dtype=np.float64)
        weekly_frame["simulated_weekly_moving_average"] = self.compute_moving_average(
            weekly_frame["simulated_weekly"],
            self.WEEKLY_MOVING_AVERAGE_WINDOW_WEEKS,
        ).to_numpy(dtype=np.float64)
        return weekly_frame

    @staticmethod
    def _resolve_fitting_start_time_week(time_grid: np.ndarray) -> float:
        """Return the model-aligned week used as the fitting window origin."""
        if len(time_grid) == 0:
            raise ValueError("ODE solution time grid is empty; cannot resolve fitting alignment.")
        return float(np.asarray(time_grid, dtype=np.float64)[0])

    def _build_solution_frame(
        self,
        time_grid: np.ndarray,
        solution_values: np.ndarray,
    ) -> pd.DataFrame:
        """Return an ODE solution dataframe with named state columns."""
        solution_array = np.asarray(solution_values, dtype=np.float64)
        if solution_array.ndim != 2:
            raise ValueError(
                "solution_values must be a 2D array with one column per ODE state; "
                f"received ndim={solution_array.ndim}."
            )
        if solution_array.shape[1] != len(self.ODE_STATE_COLUMNS):
            raise ValueError(
                "solution_values column count must match ODE_STATE_COLUMNS; "
                f"received {solution_array.shape[1]} columns."
            )

        solution_frame = pd.DataFrame(solution_array, columns=self.ODE_STATE_COLUMNS)
        solution_frame.insert(
            0,
            self.TIME_GRID_COLUMN,
            np.asarray(time_grid, dtype=np.float64),
        )
        return solution_frame

    @staticmethod
    def _calculate_padded_limits(*arrays, pad=0.1, floor=None) -> tuple[float, float]:
        """Return padded plot limits that cover the provided finite data."""
        valid_arrays = [np.asarray(array, dtype=np.float64).ravel() for array in arrays if len(array) > 0]
        if not valid_arrays:
            lower, upper = 0.0, 1.0
        else:
            stacked = np.concatenate(valid_arrays)
            finite_values = stacked[np.isfinite(stacked)]
            if finite_values.size == 0:
                lower, upper = 0.0, 1.0
            else:
                data_min = finite_values.min()
                data_max = finite_values.max()
                data_range = data_max - data_min
                pad_val = data_range * pad if data_range > 0 else max(abs(data_max) * pad, pad)
                lower = data_min - pad_val
                upper = data_max + pad_val

        if floor is not None:
            lower = max(floor, lower)
        if lower == upper:
            upper = lower + 1.0
        return lower, upper

    @staticmethod
    def _add_metric_annotation(axis, label: str, x_limits: tuple[float, float], y_limits: tuple[float, float]):
        """Place a text annotation near the top-left of the current data window."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        x_span = x_max - x_min
        y_span = y_max - y_min
        axis.text(
            x_min + 0.04 * x_span,
            y_max - 0.08 * y_span,
            label,
            fontsize=10,
            ha="left",
            va="top",
        )

    def _plot_observation_aligned_comparison_panel(
        self,
        axis,
        frame: pd.DataFrame,
        *,
        title: str,
        y_label: str,
        observed_count_column: str,
        simulated_count_column: str,
        observed_average_column: str,
        simulated_average_column: str,
        observed_count_label: str,
        simulated_count_label: str,
        observed_average_label: str,
        simulated_average_label: str,
        observed_count_color: str,
        simulated_count_color: str,
        observed_average_color: str,
        simulated_average_color: str,
        metric_label: str | None = None,
        x_limits: tuple[float, float] | None = None,
        highlight_interval: tuple[float, float] | None = None,
    ) -> None:
        """Plot observed counts, sampled simulated counts, and moving averages."""
        x_values = frame["time_week"]
        axis.plot(
            x_values,
            frame[observed_count_column],
            linestyle=":",
            marker="o",
            markersize=3.5,
            color=observed_count_color,
            fillstyle="full",
            alpha=0.75,
            label=observed_count_label,
        )
        axis.scatter(
            x_values,
            frame[simulated_count_column],
            s=14,
            color=simulated_count_color,
            alpha=0.45,
            label=simulated_count_label,
            zorder=3,
        )
        axis.plot(
            x_values,
            frame[observed_average_column],
            linestyle="--",
            linewidth=1.5,
            color=observed_average_color,
            alpha=0.9,
            label=observed_average_label,
        )
        axis.plot(
            x_values,
            frame[simulated_average_column],
            linestyle="-",
            linewidth=1.75,
            color=simulated_average_color,
            alpha=0.9,
            label=simulated_average_label,
        )

        if x_limits is None:
            x_limits = self._calculate_padded_limits(x_values, pad=0.02)
        y_limits = self._calculate_padded_limits(
            frame[observed_count_column],
            frame[simulated_count_column],
            frame[observed_average_column],
            frame[simulated_average_column],
            pad=0.15,
            floor=0.0,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        if highlight_interval is not None:
            axis.axvspan(
                highlight_interval[0],
                highlight_interval[1],
                color="lightgrey",
                alpha=0.14,
                zorder=0,
            )
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        if metric_label is not None:
            self._add_metric_annotation(axis, metric_label, x_limits, y_limits)
        axis.legend(loc="upper right", fontsize=7)

    def create_fitting_plot(self, figure=None):
        """Create the DF/DHF fitting comparison figure.
    
        Parameters
        ----------
        figure:
            Optional existing figure to clear and redraw in place. When
            provided, the same figure instance is reused for live updates.

        Returns
        -------
        matplotlib.figure.Figure
            A 2x3 subplot figure comparing observed and simulated DF/DHF data.
        
        Notes
        -----
        The figure contains six subplots:
        - Top left: Full simulated DF trajectory
        - Top middle: DF daily counts plus 7-day moving averages
        - Top right: DF weekly counts plus 3-week moving averages
        - Bottom left: Full simulated DHF trajectory
        - Bottom middle: DHF daily counts plus 7-day moving averages
        - Bottom right: DHF weekly counts plus 3-week moving averages
        """
        solution_frame = self.solution
        time_grid = solution_frame[self.TIME_GRID_COLUMN].to_numpy()
        Y_m1_h = solution_frame["Y_m1_h"].to_numpy()
        z = solution_frame["z"].to_numpy()
        df_plot_frame, dhf_plot_frame = self._build_plot_comparison_frames()
        weekly_df_plot_frame = self._build_weekly_fitting_frame(df_plot_frame)
        weekly_dhf_plot_frame = self._build_weekly_fitting_frame(dhf_plot_frame)
        plot_start_week, plot_end_week = self._resolve_plot_time_bounds(time_grid)
        error_start_week, error_end_week = self._resolve_error_time_bounds(time_grid)
        plot_x_limits = (plot_start_week, plot_end_week)
        error_highlight = (error_start_week, error_end_week)

        if figure is None:
            figure, axes = plt.subplots(
                2,
                3,
                figsize=self.FITTING_PLOT_FIGSIZE,
                constrained_layout=False,
            )
        else:
            figure.set_size_inches(*self.FITTING_PLOT_FIGSIZE, forward=True)
            if len(figure.axes) == 6:
                axes = np.asarray(figure.axes, dtype=object).reshape(2, 3)
                for axis in axes.flat:
                    axis.cla()
            else:
                figure.clear()
                axes = figure.subplots(2, 3)
        if hasattr(figure, "set_layout_engine"):
            figure.set_layout_engine(None)
        figure.subplots_adjust(**self.FITTING_PLOT_LAYOUT)

        # Left column: qualitative view of the full simulated series.
        df_left_ymin, df_left_ymax = self._calculate_padded_limits(
            z,
            pad=0.15,
            floor=0.0,
        )
        dhf_left_ymin, dhf_left_ymax = self._calculate_padded_limits(
            Y_m1_h,
            pad=0.15,
            floor=0.0,
        )

        axes[0, 0].plot(
            time_grid, z,
            '--'
        )
        axes[0, 0].set_title(r'Estimated Number of Reported DF  cases')
        axes[0, 0].set_xlim(*plot_x_limits)
        axes[0, 0].set_ylim(df_left_ymin, df_left_ymax)
        axes[0, 0].grid(alpha=0.3, linestyle='--', linewidth=0.5)

        self._plot_observation_aligned_comparison_panel(
            axes[0, 1],
            df_plot_frame,
            title='DF Daily Counts and 7-day Average',
            y_label='Daily cases',
            observed_count_column='observed_daily',
            simulated_count_column='simulated_daily',
            observed_average_column='observed_moving_average',
            simulated_average_column='simulated_moving_average',
            observed_count_label='Observed daily counts',
            simulated_count_label='Sampled simulated daily counts',
            observed_average_label='Observed 7-day average',
            simulated_average_label='Simulated 7-day average',
            observed_count_color='lightskyblue',
            simulated_count_color='navy',
            observed_average_color='steelblue',
            simulated_average_color='darkblue',
            metric_label='err=' + str(np.round(self.df_fit_error, 1)),
            x_limits=plot_x_limits,
            highlight_interval=error_highlight,
        )
        self._plot_observation_aligned_comparison_panel(
            axes[0, 2],
            weekly_df_plot_frame,
            title='DF Weekly Counts and 3-week Average',
            y_label='Weekly cases',
            observed_count_column='observed_weekly',
            simulated_count_column='simulated_weekly',
            observed_average_column='observed_weekly_moving_average',
            simulated_average_column='simulated_weekly_moving_average',
            observed_count_label='Observed weekly counts',
            simulated_count_label='Sampled simulated weekly counts',
            observed_average_label='Observed 3-week average',
            simulated_average_label='Simulated 3-week average',
            observed_count_color='paleturquoise',
            simulated_count_color='teal',
            observed_average_color='cadetblue',
            simulated_average_color='darkslateblue',
            metric_label='err=' + str(np.round(self.df_weekly_fit_error, 1)),
            x_limits=plot_x_limits,
            highlight_interval=error_highlight,
        )

        axes[1, 0].plot(time_grid, Y_m1_h, 'r-')
        axes[1, 0].set_title(r'Estimated Number of Reported DHF cases')
        axes[1, 0].set_xlim(*plot_x_limits)
        axes[1, 0].set_ylim(dhf_left_ymin, dhf_left_ymax)
        axes[1, 0].grid(alpha=0.3, linestyle='--', linewidth=0.5)

        self._plot_observation_aligned_comparison_panel(
            axes[1, 1],
            dhf_plot_frame,
            title='DHF Daily Counts and 7-day Average',
            y_label='Daily cases',
            observed_count_column='observed_daily',
            simulated_count_column='simulated_daily',
            observed_average_column='observed_moving_average',
            simulated_average_column='simulated_moving_average',
            observed_count_label='Observed daily counts',
            simulated_count_label='Sampled simulated daily counts',
            observed_average_label='Observed 7-day average',
            simulated_average_label='Simulated 7-day average',
            observed_count_color='moccasin',
            simulated_count_color='firebrick',
            observed_average_color='darkorange',
            simulated_average_color='crimson',
            metric_label='err=' + str(np.round(self.dhf_fit_error, 1)),
            x_limits=plot_x_limits,
            highlight_interval=error_highlight,
        )
        self._plot_observation_aligned_comparison_panel(
            axes[1, 2],
            weekly_dhf_plot_frame,
            title='DHF Weekly Counts and 3-week Average',
            y_label='Weekly cases',
            observed_count_column='observed_weekly',
            simulated_count_column='simulated_weekly',
            observed_average_column='observed_weekly_moving_average',
            simulated_average_column='simulated_weekly_moving_average',
            observed_count_label='Observed weekly counts',
            simulated_count_label='Sampled simulated weekly counts',
            observed_average_label='Observed 3-week average',
            simulated_average_label='Simulated 3-week average',
            observed_count_color='peachpuff',
            simulated_count_color='brown',
            observed_average_color='sandybrown',
            simulated_average_color='darkred',
            metric_label='err=' + str(np.round(self.dhf_weekly_fit_error, 1)),
            x_limits=plot_x_limits,
            highlight_interval=error_highlight,
        )

        for row_index in np.arange(2):
            for column_index in np.arange(3):
                axes[row_index, column_index].set(xlabel='week n')
                axes[row_index, column_index].xaxis.set_label_coords(0.5, -0.14)
        axes[0, 0].set(ylabel='Individuals')
        axes[1, 0].set(ylabel='Individuals')

        return figure
    
    def save_fitting_plot(self):
        """Save the DF/DHF fitting comparison figure to disk.

        Creates a 2x3 subplot figure comparing the full simulated DF/DHF
        trajectories plus the daily and weekly observation-aligned fitting
        panels, then saves it to the plots directory.
        The file is saved as ``fitting_DF_DHF.png`` in ``self.plots_dir``.
        """
        figure = self.create_fitting_plot()
        figure.savefig(self.plots_dir / 'fitting_DF_DHF.png')
        plt.close(figure)

    def create_reference_parameter_fitting_plot(self, figure=None):
        """Create a fitting plot from the reference parameter set.

        This method loads the default reference parameters from
        ``data/default_model_parameters.json`` (via deterministic sampling),
        solves the ODE system, computes fitting errors, and then renders the
        fitting plot.

        Parameters
        ----------
        figure:
            Optional existing figure to redraw in place.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered fitting figure using the reference parameters.
        """
        self.sample_model_parameters(flag_deterministic=True)
        self.solve_ode_system()
        self.compute_fitting_errors()
        reference_plot = self.create_fitting_plot(figure=figure)
        return reference_plot

    def save_input_data_plot(self):
        """Plot weekly DF and DHF frequency tables in two stacked axes.

        The underlying weekly CSVs are regenerated through the shared
        aggregation pipeline so the plots, dataframes, and persisted tables all
        use the same counting logic.
        """
        weekly_df_counts, weekly_dhf_counts = self.build_weekly_frequency_tables()

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 6),
            sharex=True
        )

        axes[0].plot(
            weekly_df_counts['week'],
            weekly_df_counts['count'],
            linestyle='', marker='o', markersize=4,
            color='steelblue', alpha=0.6
        )
        axes[0].set_title('Incidence of DF cases per week')
        axes[0].set_ylabel('Dengue Fever incidence')

        axes[1].plot(
            weekly_dhf_counts['week'],
            weekly_dhf_counts['count'],
            linestyle='', marker='o', markersize=4,
            color='tomato',
            alpha=0.6
        )
        axes[1].set_title('DHF cases over time')
        axes[1].set_ylabel('Dengue Hemorrhagic Fever incidence')
        axes[1].set_xlabel('Date')

        for axis, weekly_counts in zip(axes, (weekly_df_counts, weekly_dhf_counts)):
            axis.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            y_min, y_max = self._calculate_padded_limits(weekly_counts['count'], pad=0.1, floor=0.0)
            axis.set_ylim(y_min, y_max)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'input_cases_timeseries.png')
        plt.close(fig)

    def compute_fitting_errors(self):
        """Compute the current daily and weekly DF/DHF moving-average fit errors."""
        solution_frame = self.solution
        z = solution_frame["z"].to_numpy()
        self.peak_df_cases = np.max(z)
        df_fitting_frame, dhf_fitting_frame = self._build_fitting_comparison_frames()
        weekly_df_fitting_frame = self._build_weekly_fitting_frame(df_fitting_frame)
        weekly_dhf_fitting_frame = self._build_weekly_fitting_frame(dhf_fitting_frame)
        self.df_fit_error = self._compute_moving_average_fit_error(df_fitting_frame)
        self.dhf_fit_error = self._compute_moving_average_fit_error(dhf_fitting_frame)
        self.df_weekly_fit_error = self._compute_moving_average_fit_error(
            weekly_df_fitting_frame,
            observed_column="observed_weekly_moving_average",
            simulated_column="simulated_weekly_moving_average",
        )
        self.dhf_weekly_fit_error = self._compute_moving_average_fit_error(
            weekly_dhf_fitting_frame,
            observed_column="observed_weekly_moving_average",
            simulated_column="simulated_weekly_moving_average",
        )

    @staticmethod
    def compute_ode_rhs(
            state_vector,
            _time,
            *ode_args,
            q=None,
            **params,
    ) -> np.ndarray:
        """Evaluate the ODE right-hand side from equations (3.1) to (3.3).

        The extra ``z`` state kept by the solver is interpreted as reported
        classical-incidence. That incidence term is inferred from the
        cumulative-incidence equations introduced immediately after (3.3), so
        the public solver interface remains unchanged.
        """
        del q  # Retained only for backward compatibility with older call sites.

        if ode_args:
            if len(ode_args) != len(StochasticSearch.ODE_ARGUMENT_NAMES):
                raise TypeError(
                    "compute_ode_rhs expected "
                    f"{len(StochasticSearch.ODE_ARGUMENT_NAMES)} ODE parameters, "
                    f"received {len(ode_args)}."
                )
            params = {
                **dict(zip(StochasticSearch.ODE_ARGUMENT_NAMES, ode_args)),
                **params,
            }

        state_vector = np.asarray(state_vector, dtype=np.float64)

        (
            M_s, M_I1, M_I2,
            S, I_1, I_2, R_s,
            S_m1, Y_m1_c, Y_m1_h, R_s_m1,
            z,
        ) = state_vector
        (
            Lambda_M,
            Lambda_S,
            Lambda_S_m1,
            beta_M,
            beta_H,
            b,
            mu_M,
            mu_H,
            alpha_c,
            alpha_h,
            sigma,
            p,
            theta,
        ) = operator.itemgetter(*StochasticSearch.ODE_ARGUMENT_NAMES)(params)

        primary_host_population = Lambda_S / mu_H
        secondary_host_population = Lambda_S_m1 / mu_H
        N_H = primary_host_population + secondary_host_population

        vector_to_host_force = beta_H * b / N_H
        host_to_vector_force = beta_M * b / N_H
        A_I1 = host_to_vector_force * I_1
        A_I2 = host_to_vector_force * I_2
        A_Y_m1_c = host_to_vector_force * Y_m1_c
        A_Y_m1_h = host_to_vector_force * Y_m1_h
        A_total = A_I1 + A_I2 + A_Y_m1_c + A_Y_m1_h
        B_M1 = vector_to_host_force * M_I1
        B_M2 = vector_to_host_force * M_I2
        reinfection_incidence = sigma * B_M2 * S_m1

        dM_s = Lambda_M - A_total * M_s - mu_M * M_s
        dM_I1 = A_I1 * M_s - mu_M * M_I1
        dM_I2 = (A_I2 + A_Y_m1_c + A_Y_m1_h) * M_s - mu_M * M_I2

        dS = Lambda_S - (B_M1 + B_M2) * S - mu_H * S
        dI_1 = B_M1 * S - (alpha_c + mu_H) * I_1
        dI_2 = B_M2 * S - (alpha_c + mu_H) * I_2
        dR_s = alpha_c * (I_1 + I_2 ) - mu_H * R_s

        dS_m1 = Lambda_S_m1 - reinfection_incidence - mu_H * S_m1
        dY_m1_c = (1.0 - theta) * reinfection_incidence - (alpha_c + mu_H) * Y_m1_c
        dY_m1_h = theta * reinfection_incidence - (alpha_h + mu_H) * Y_m1_h
        dR_s_m1 = alpha_c * Y_m1_c + alpha_h * Y_m1_h - mu_H * R_s_m1
        dz = p * (dI_1 + dI_2 + dY_m1_c)
#
        rhs = np.array(
            [
                dM_s,
                dM_I1,
                dM_I2,
                dS,
                dI_1,
                dI_2,
                dR_s,
                dS_m1,
                dY_m1_c,
                dY_m1_h,
                dR_s_m1,
                dz
            ],
            dtype=np.float64
        )
        return rhs
#
    def solve_ode_system(self):
        """Integrate the model ODE system over the configured time grid.

        Returns
        -------
        pandas.DataFrame
            A solution table with ``time_grid`` plus one named column per
            compartment/state in ``ODE_STATE_COLUMNS``.
        """
        final_time = self.T
        initial_time = self.t0
        time_grid = np.linspace(initial_time, final_time, self.grid_size)
        initial_state = np.array(
            [self.M_s0, self.M_10, self.M_20,
             self.S_0, self.I_10, self.I_20, self.R_s0,
             self.S_m1_0, self.Y_m1_c0, self.Y_m1_h0, self.R_s_m1_0,
             self.z0])

        Lambda_M = self.Lambda_M
        Lambda_S = self.Lambda_S
        Lambda_S_m1 = self.Lambda_S_m1
        beta_M = self.beta_M
        beta_H = self.beta_H
        b = self.b
        mu_M = self.mu_M
        mu_H = self.mu_H
        alpha_c = self.alpha_c
        alpha_h = self.alpha_h
        sigma = self.sigma
        p = self.p
        theta = self.theta

        solution_values = integrate.odeint(
            self.compute_ode_rhs,
            initial_state,
            time_grid,
            args=(
                Lambda_M,
                Lambda_S,
                Lambda_S_m1,
                beta_M,
                beta_H,
                b,
                mu_M,
                mu_H,
                alpha_c,
                alpha_h,
                sigma,
                p,
                theta,
            ),
        )
        solution_frame = self._build_solution_frame(time_grid, solution_values)
        self.solution = solution_frame
        self.t = time_grid
        return solution_frame

    def _build_sample_parameter_frame(self) -> pd.DataFrame:
        """Return the sampled-parameter state as a single-row dataframe."""
        return pd.DataFrame(
            [{column: getattr(self, column) for column in self.SAMPLED_PARAMETER_COLUMNS}]
        )

    def _load_default_parameter_map(self) -> dict:
        """Load and flatten the default grouped model-parameter JSON payload."""
        default_parameter_path = get_repository_root() / self.DEFAULT_MODEL_PARAMETER_FILE
        with default_parameter_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self._flatten_parameter_payload(payload)

    def _refresh_derived_population_constraints(self):
        """Refresh derived population totals and recruitment-rate identities."""
        self.N_S = self.S_0 + self.I_10 + self.I_20 + self.R_s0
        self.N_S_m1 = self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0 + self.R_s_m1_0
        # Backward-compatible aliases retained for legacy call sites.
        self.N_H = self.N_S
        self.N_sm1 = self.N_S_m1
        self.Lambda_S = self.mu_H * self.N_S
        self.Lambda_S_m1 = self.mu_H * self.N_S_m1
        self.Lambda_M = self.mu_M * (self.M_s0 + self.M_10 + self.M_20)
        self.z0 = self.p * (self.I_10 + self.I_20 + self.Y_m1_c0)

    def sample_model_parameters(self, flag_deterministic=False):
        """Sample a new parameter set and update the model state in place.

        Parameters
        ----------
        flag_deterministic:
            When ``True``, load a fixed baseline parameter set instead of
            drawing random values.

        Returns
        -------
        pandas.DataFrame
            A one-row dataframe containing the sampled parameter values in a
            stable column order.
        """
        if flag_deterministic:
            default_parameter_path = get_repository_root() / self.DEFAULT_MODEL_PARAMETER_FILE
            self.load_parameters_from_json(default_parameter_path, strict=True)
        else:
            # PDF-guided stochastic proposal:
            # - Table 2 ranges for transmission and vital-dynamics rates.
            # - Demographic identities from model (3.3):
            #   Lambda_S = mu_H * N_S, Lambda_S_m1 = mu_H * N_S_m1,
            #   Lambda_M = mu_M * N_M.
            default_params = self._load_default_parameter_map()
            machine_eps = np.finfo(np.float64).eps

            # Vital/transmission rates (Table 2 in DengueProyect.pdf).
            self.b = np.random.uniform(10.36, 33.39)
            self.mu_M = np.random.uniform(0.252, 0.763)
            self.mu_H = np.float64(default_params["mu_H"])
            self.beta_H = np.random.uniform(machine_eps, 0.05)
            self.beta_M = np.random.uniform(machine_eps, 0.05)
            self.alpha_c = np.random.uniform(0.581, 1.75)
            self.alpha_h = np.random.uniform(0.581, 1.75)
            self.sigma = np.random.uniform(machine_eps, 5.0)
            self.p = np.random.uniform(1.0 / 60.0, 1.0 / 30.0)

            theta_mean = float(default_params.get("theta", 0.05))
            theta_mean = np.clip(theta_mean, machine_eps, 1.0 - machine_eps)
            theta_concentration = 40.0
            theta_alpha = theta_mean * theta_concentration
            theta_beta = (1.0 - theta_mean) * theta_concentration
            self.theta = np.random.beta(theta_alpha, theta_beta)

            # Keep total host population near the deterministic baseline and
            # sample the secondary-susceptible share around its baseline value.
            total_host_population = float(default_params["N_S"] + default_params["N_S_m1"])
            baseline_secondary_share = float(default_params["N_S_m1"]) / total_host_population
            sampled_secondary_share = np.random.normal(
                loc=baseline_secondary_share,
                scale=0.0025,
            )
            sampled_secondary_share = np.clip(sampled_secondary_share, 0.005, 0.20)
            N_S_m1 = total_host_population * sampled_secondary_share
            N_S = total_host_population - N_S_m1

            self.I_10 = max(1.0, float(np.random.poisson(lam=max(1.0, float(default_params["I_10"])))))
            self.I_20 = max(1.0, float(np.random.poisson(lam=max(1.0, float(default_params["I_20"])))))
            self.R_s0 = 0.0
            self.S_0 = max(0.0, N_S - (self.I_10 + self.I_20 + self.R_s0))

            self.Y_m1_c0 = float(np.random.poisson(lam=0.5))
            self.Y_m1_h0 = float(np.random.poisson(lam=max(1.0, float(default_params["Y_m1_h0"]))))
            self.R_s_m1_0 = 0.0
            secondary_infected_total = self.Y_m1_c0 + self.Y_m1_h0 + self.R_s_m1_0
            if secondary_infected_total >= N_S_m1:
                scale = (0.9 * N_S_m1) / max(secondary_infected_total, machine_eps)
                self.Y_m1_c0 *= scale
                self.Y_m1_h0 *= scale
            self.S_m1_0 = max(0.0, N_S_m1 - (self.Y_m1_c0 + self.Y_m1_h0 + self.R_s_m1_0))

            total_vector_population = float(
                default_params.get(
                    "N_M",
                    default_params["M_s0"] + default_params["M_10"] + default_params["M_20"],
                )
            )
            baseline_infected_vectors = float(default_params["M_10"] + default_params["M_20"])
            infected_vectors = max(
                2.0,
                baseline_infected_vectors * np.random.uniform(0.5, 1.5),
            )
            infected_vectors = min(infected_vectors, 0.5 * total_vector_population)
            primary_vector_share = np.random.uniform(0.35, 0.65)
            self.M_10 = infected_vectors * primary_vector_share
            self.M_20 = infected_vectors - self.M_10
            self.M_s0 = max(0.0, total_vector_population - infected_vectors)

            self.Rec_0 = 0.0

        self._refresh_derived_population_constraints()
        self.h = np.float64(self.T) / np.float64(self.grid_size)
        sampled_parameter_frame = self._build_sample_parameter_frame()
        return sampled_parameter_frame

    def _build_parameter_snapshot_dict(self) -> dict:
        """Return the current parameter and metric state as JSON-serializable scalars."""
        return {
            'Lambda_M': float(self.Lambda_M),
            'Lambda_S': float(self.Lambda_S),
            'Lambda_S_m1': float(self.Lambda_S_m1),
            'beta_M': float(self.beta_M),
            'beta_H': float(self.beta_H),
            'b': float(self.b),
            'mu_M': float(self.mu_M),
            'mu_H': float(self.mu_H),
            'alpha_c': float(self.alpha_c),
            'alpha_h': float(self.alpha_h),
            'sigma': float(self.sigma),
            'p': float(self.p),
            'theta': float(self.theta),
            'M_s0': float(self.M_s0),
            'M_10': float(self.M_10),
            'M_20': float(self.M_20),
            'S_0': float(self.S_0),
            'I_10': float(self.I_10),
            'I_20': float(self.I_20),
            'S_m1_0': float(self.S_m1_0),
            'Y_m1_c0': float(self.Y_m1_c0),
            'Y_m1_h0': float(self.Y_m1_h0),
            'R_s0': float(self.R_s0),
            'R_s_m1_0': float(self.R_s_m1_0),
            'Rec_0': float(self.Rec_0),
            'z0': float(self.z0),
            't0': float(self.t0),
            'h': float(self.h),
            'T': float(self.T),
            'grid_size': int(self.grid_size),
            'N_H': float(self.N_S),
            'N_sm1': float(self.N_S_m1),
            'r_01': float(self.r_01),
            'r_02': float(self.r_02),
            'r_zero': float(self.r_zero),
            'df_fit_error': float(self.df_fit_error),
            'dhf_fit_error': float(self.dhf_fit_error),
            'df_weekly_fit_error': float(self.df_weekly_fit_error),
            'dhf_weekly_fit_error': float(self.dhf_weekly_fit_error),
            'peak_df_cases': float(self.peak_df_cases),
            'meets_r_zero_threshold': bool(self.meets_r_zero_threshold),
            'meets_df_error_threshold': bool(self.meets_df_error_threshold),
            'meets_dhf_error_threshold': bool(self.meets_dhf_error_threshold),
            'meets_acceptance_criteria': bool(self.meets_acceptance_criteria),
        }

    def save_parameter_snapshot(self, file_name_prefix=None, file_path=None):
        """Persist the current parameter state as a JSON snapshot."""
        parameters = self._build_parameter_snapshot_dict()
        if file_path is not None:
            output_path = Path(file_path)
        else:
            str_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            prefix = file_name_prefix or str(self.output_dir / 'parameters_')
            output_path = Path(f"{prefix}{str_time}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as outfile:
            json.dump(parameters, outfile, indent=2)
        return output_path

    def save_solution_snapshot(self, file_path) -> Path:
        """Persist the current ODE solution table as CSV."""
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.solution.to_csv(output_path, index=False)
        return output_path

    def build_fitting_scale_solution_frame(self) -> pd.DataFrame:
        """Return solution-derived fitting data on the same scale used by plots."""
        df_plot_frame, dhf_plot_frame = self._build_plot_comparison_frames()
        weekly_df_plot_frame = self._build_weekly_fitting_frame(df_plot_frame)
        weekly_dhf_plot_frame = self._build_weekly_fitting_frame(dhf_plot_frame)

        def build_daily_records(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
            records = frame.reset_index().rename(
                columns={
                    "date": "date",
                    "observed_daily": "observed_count",
                    "simulated_daily": "simulated_count",
                    "observed_moving_average": "observed_moving_average",
                    "simulated_moving_average": "simulated_moving_average",
                }
            )
            records.insert(0, "outcome", outcome)
            records.insert(0, "frequency", "daily")
            records["moving_average_window"] = self.MOVING_AVERAGE_WINDOW_DAYS
            return records[
                [
                    "frequency",
                    "outcome",
                    "date",
                    "time_week",
                    "observed_count",
                    "simulated_count",
                    "observed_moving_average",
                    "simulated_moving_average",
                    "moving_average_window",
                ]
            ]

        def build_weekly_records(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
            records = frame.rename(
                columns={
                    "week": "date",
                    "observed_weekly": "observed_count",
                    "simulated_weekly": "simulated_count",
                    "observed_weekly_moving_average": "observed_moving_average",
                    "simulated_weekly_moving_average": "simulated_moving_average",
                }
            ).copy()
            records.insert(0, "outcome", outcome)
            records.insert(0, "frequency", "weekly")
            records["moving_average_window"] = self.WEEKLY_MOVING_AVERAGE_WINDOW_WEEKS
            return records[
                [
                    "frequency",
                    "outcome",
                    "date",
                    "time_week",
                    "observed_count",
                    "simulated_count",
                    "observed_moving_average",
                    "simulated_moving_average",
                    "moving_average_window",
                ]
            ]

        fitting_scale_frame = pd.concat(
            [
                build_daily_records(df_plot_frame, "DF"),
                build_daily_records(dhf_plot_frame, "DHF"),
                build_weekly_records(weekly_df_plot_frame, "DF"),
                build_weekly_records(weekly_dhf_plot_frame, "DHF"),
            ],
            ignore_index=True,
        )
        fitting_scale_frame["date"] = pd.to_datetime(fitting_scale_frame["date"]).dt.strftime("%Y-%m-%d")
        return fitting_scale_frame

    def save_fitting_scale_solution(self, file_path) -> Path:
        """Persist the plot-scale fitting data for the current solution as CSV."""
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.build_fitting_scale_solution_frame().to_csv(output_path, index=False)
        return output_path

    def build_sampled_solution_time_series(self, sample_interval_days: int = 1) -> pd.DataFrame:
        """Return the current ODE solution sampled onto a day-based time axis.

        The native solver state is stored on a dense weekly grid. This helper
        interpolates each state onto evenly-spaced daily samples so downstream
        analysis can handle the solution as a regular time series.
        """
        if sample_interval_days <= 0:
            raise ValueError("sample_interval_days must be a positive integer.")

        solution_frame = self.solution
        time_weeks = solution_frame[self.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
        if time_weeks.size == 0:
            return pd.DataFrame(columns=[self.TIME_DAY_COLUMN, *self.ODE_SOLUTION_COLUMNS])

        start_week = float(time_weeks[0])
        end_week = float(time_weeks[-1])
        total_days = int(round((end_week - start_week) * self.DAYS_PER_WEEK))
        sampled_days = np.arange(0, total_days + 1, sample_interval_days, dtype=int)
        if sampled_days[-1] != total_days:
            sampled_days = np.append(sampled_days, total_days)

        sampled_weeks = start_week + (
            sampled_days.astype(np.float64) / self.DAYS_PER_WEEK
        )
        sampled_solution = {
            self.TIME_DAY_COLUMN: sampled_days,
            self.TIME_GRID_COLUMN: sampled_weeks,
        }
        for column in self.ODE_STATE_COLUMNS:
            sampled_solution[column] = np.interp(
                sampled_weeks,
                time_weeks,
                solution_frame[column].to_numpy(dtype=np.float64),
            )
        return pd.DataFrame(sampled_solution)

    def save_sampled_solution_time_series(
        self,
        file_path=None,
        sample_interval_days: int = 1,
    ) -> Path:
        """Persist a day-sampled solution time series as CSV."""
        output_path = (
            Path(file_path)
            if file_path is not None
            else self.output_dir / "solution_time_series.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_solution = self.build_sampled_solution_time_series(
            sample_interval_days=sample_interval_days,
        )
        sampled_solution.to_csv(output_path, index=False)
        return output_path

    def load_solution_snapshot(self, file_path) -> pd.DataFrame:
        """Load a previously-saved ODE solution CSV into the current instance."""
        solution_frame = pd.read_csv(file_path)
        expected_columns = list(self.ODE_SOLUTION_COLUMNS)
        if list(solution_frame.columns) != expected_columns:
            raise ValueError(
                "Solution snapshot columns must match ODE_SOLUTION_COLUMNS; "
                f"received {list(solution_frame.columns)}."
            )
        solution_frame = solution_frame.astype(np.float64)
        self.solution = solution_frame
        self.t = solution_frame[self.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
        self.grid_size = len(solution_frame.index)
        if self.grid_size > 1:
            self.h = np.float64(self.t[1] - self.t[0])
        return solution_frame

    def _serialize_fitting_frame(self, fitting_frame: pd.DataFrame) -> list[dict]:
        """Return a fitting comparison frame as JSON-friendly records."""
        serialized_frame = fitting_frame.reset_index().copy()
        serialized_frame["date"] = serialized_frame["date"].dt.strftime("%Y-%m-%d")
        return serialized_frame.to_dict(orient="records")

    def _build_acceptance_snapshot_dict(
        self,
        sample_index=None,
        snapshot_timestamp=None,
        parameter_file=None,
        solution_file=None,
        fitting_scale_solution_file=None,
        solution_time_series_file=None,
        fitting_plot_file=None,
        populations_plot_file=None,
    ) -> dict:
        """Build a reproducibility bundle for the current accepted state."""
        r01_per_week, r02_per_week, r0_per_week = self.compute_basic_reproduction_numbers()
        accepted = self.evaluate_search_acceptance()
        df_fitting_frame, dhf_fitting_frame = self._build_fitting_comparison_frames()

        def normalize_path(path):
            if path is None:
                return None
            path = Path(path)
            try:
                return str(path.relative_to(self.runtime_dir))
            except ValueError:
                return str(path)

        fitting_start_time_week, fitting_end_time_week = self._resolve_error_time_bounds(
            self.solution[self.TIME_GRID_COLUMN].to_numpy(dtype=np.float64)
        )
        daily_df_counts, _ = self._load_daily_frequency_tables_for_fitting()
        fitting_start_date = self._get_iso_week_start_date(
            daily_df_counts,
            fitting_start_time_week,
        ).strftime("%Y-%m-%d")

        return {
            "created_at": snapshot_timestamp or datetime.datetime.now().isoformat(timespec="seconds"),
            "sample_index": None if sample_index is None else int(sample_index),
            "data_dir": str(self.data_dir),
            "runtime_dir": str(self.runtime_dir),
            "parameters": self._build_parameter_snapshot_dict(),
            "acceptance_thresholds": {
                "r_zero_min": float(self.R_ZERO_ACCEPTANCE_THRESHOLD),
                "df_fit_error_max": float(self.df_error_threshold),
                "dhf_fit_error_max": float(self.dhf_error_threshold),
                "peak_df_cases_max": float(self.PEAK_DF_CASES_THRESHOLD),
                "fitting_window_weeks": int(self.FITTING_WINDOW_WEEKS),
                "moving_average_window_days": int(self.MOVING_AVERAGE_WINDOW_DAYS),
                "df_fitting_start_index": int(self.DF_FITTING_START_INDEX),
                "dhf_fitting_start_index": int(self.DHF_FITTING_START_INDEX),
                "df_fitting_start_week": float(fitting_start_time_week),
                "dhf_fitting_start_week": float(fitting_start_time_week),
                "fitting_start_week": float(fitting_start_time_week),
                "fitting_end_week": float(fitting_end_time_week),
                "plot_end_week": float(self.PLOT_END_WEEK),
                "fitting_start_date": fitting_start_date,
            },
            "acceptance_metrics": {
                "r01_per_week": float(r01_per_week),
                "r02_per_week": float(r02_per_week),
                "r0_per_week": float(r0_per_week),
                "df_fit_error": float(self.df_fit_error),
                "dhf_fit_error": float(self.dhf_fit_error),
                "df_weekly_fit_error": float(self.df_weekly_fit_error),
                "dhf_weekly_fit_error": float(self.dhf_weekly_fit_error),
                "peak_df_cases": float(self.peak_df_cases),
                "meets_r_zero_threshold": bool(self.meets_r_zero_threshold),
                "meets_df_error_threshold": bool(self.meets_df_error_threshold),
                "meets_dhf_error_threshold": bool(self.meets_dhf_error_threshold),
                "meets_acceptance_criteria": bool(accepted),
            },
            "artifacts": {
                "parameter_snapshot": normalize_path(parameter_file),
                "solution_snapshot": normalize_path(solution_file),
                "fitting_scale_solution": normalize_path(fitting_scale_solution_file),
                "solution_time_series": normalize_path(solution_time_series_file),
                "fitting_plot": normalize_path(fitting_plot_file),
                "populations_plot": normalize_path(populations_plot_file),
            },
            "df_fitting_window": self._serialize_fitting_frame(df_fitting_frame),
            "dhf_fitting_window": self._serialize_fitting_frame(dhf_fitting_frame),
        }

    def save_acceptance_snapshot(
        self,
        sample_index=None,
        snapshot_timestamp=None,
        parameter_file=None,
        solution_file=None,
        fitting_scale_solution_file=None,
        solution_time_series_file=None,
        fitting_plot_file=None,
        populations_plot_file=None,
    ) -> Path:
        """Persist a reproducibility bundle for the current accepted state."""
        timestamp = snapshot_timestamp or datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        parameter_path = (
            Path(parameter_file)
            if parameter_file is not None
            else self.output_dir / f"parameters_{timestamp}.json"
        )
        solution_path = (
            Path(solution_file)
            if solution_file is not None
            else self.output_dir / f"solution_{timestamp}.csv"
        )
        fitting_scale_solution_path = (
            Path(fitting_scale_solution_file)
            if fitting_scale_solution_file is not None
            else self.output_dir / f"solution_fitting_scale_{timestamp}.csv"
        )
        solution_time_series_path = (
            Path(solution_time_series_file)
            if solution_time_series_file is not None
            else self.output_dir / f"solution_time_series_{timestamp}.csv"
        )
        self.save_parameter_snapshot(file_path=parameter_path)
        self.save_solution_snapshot(solution_path)
        self.save_fitting_scale_solution(fitting_scale_solution_path)
        self.save_sampled_solution_time_series(file_path=solution_time_series_path)
        snapshot_path = self.output_dir / f"acceptance_snapshot_{timestamp}.json"
        snapshot = self._build_acceptance_snapshot_dict(
            sample_index=sample_index,
            snapshot_timestamp=timestamp,
            parameter_file=parameter_path,
            solution_file=solution_path,
            fitting_scale_solution_file=fitting_scale_solution_path,
            solution_time_series_file=solution_time_series_path,
            fitting_plot_file=fitting_plot_file,
            populations_plot_file=populations_plot_file,
        )
        with snapshot_path.open('w', encoding='utf-8') as outfile:
            json.dump(snapshot, outfile, indent=2)
        return snapshot_path

    def compute_basic_reproduction_numbers(self):
        """Compute the basic reproduction number and its two components."""
        # load parameters
        Lambda_M = self.Lambda_M
        beta_M = self.beta_M
        beta_H = self.beta_H
        b = self.b
        mu_M = self.mu_M
        mu_H = self.mu_H
        alpha_c = self.alpha_c
        alpha_h = self.alpha_h
        sigma = self.sigma
        theta = self.theta
        S_0 = self.S_0
        S_m1_0 = self.S_m1_0
        #
        p = self.p
        theta = self.theta
        #
        N_H = self.N_S
        N_sm1 = self.N_S_m1
        #
        #
        #
        pi_r = (beta_H * beta_M * b ** 2 * Lambda_M) / (N_H ** 2 * mu_M ** 2)
        r_01 = pi_r * ((N_H - N_sm1) + + sigma * (1.0 - theta) * N_sm1) \
               * (alpha_c + mu_H) ** (-1)
        #
        #
        #
        r_02 = pi_r * sigma * theta * N_sm1 * (alpha_c + alpha_h) ** (-1)
        #
        #
        #
        self.r_01 = r_01
        self.r_02 = r_02
        r_zero = np.sqrt(self.r_01 + self.r_02)
        self.r_zero = r_zero
        return np.sqrt(r_01), np.sqrt(r_02), r_zero

    def save_solution_plots(self):
        """Write compartment population plots for the current ODE solution."""
        solution_frame = self.solution
        M_s = solution_frame["M_s"].to_numpy()
        M_1 = solution_frame["M_I1"].to_numpy()
        M_2 = solution_frame["M_I2"].to_numpy()
        S = solution_frame["S"].to_numpy()
        I_1 = solution_frame["I_1"].to_numpy()
        I_2 = solution_frame["I_2"].to_numpy()
        R_s = solution_frame["R_s"].to_numpy()
        S_m1 = solution_frame["S_m1"].to_numpy()
        Y_m1_c = solution_frame["Y_m1_c"].to_numpy()
        Y_m1_h = solution_frame["Y_m1_h"].to_numpy()
        R_s_m1 = solution_frame["R_s_m1"].to_numpy()
        z = solution_frame["z"].to_numpy()
        recovers = R_s + R_s_m1
        #
        t = solution_frame[self.TIME_GRID_COLUMN].to_numpy()
        N_H = S + I_1 + I_2 + R_s + S_m1 + Y_m1_c + Y_m1_h + R_s_m1
        #
        f1, ax_array = plt.subplots(4, 3, sharex=True)
        #
        ax_array[0, 0].plot(t, M_s)
        ax_array[0, 0].set_title(r'$M_s$')
        #
        #
        ax_array[0, 1].plot(t, M_1)
        ax_array[0, 1].set_title(r'$M_1$')
        #
        #
        ax_array[0, 2].plot(t, M_2)
        ax_array[0, 2].set_title(r'$M_2$')
        #
        #   Infected humans first time
        ax_array[1, 0].plot(t, S)
        ax_array[1, 0].set_title(r'$I_s$')
        #
        #
        ax_array[1, 1].plot(t, I_1)
        ax_array[1, 1].set_title(r'$I_1$')
        #
        #
        ax_array[1, 2].plot(t, I_2)
        ax_array[1, 2].set_title(r'$I_2$')
        ##
        ##
        ax_array[2, 0].plot(t, S_m1)
        ax_array[2, 0].set_title(r'$S_{-1}$')
        ax_array[2, 1].plot(t, Y_m1_c)
        ax_array[2, 1].set_title(r'$Y_{-1}^{[c]}$')
        #
        #
        ax_array[2, 2].plot(t, Y_m1_h)
        ax_array[2, 2].set_title(r'$Y_{-1}^{[h]}$')
        #
        #
        ax_array[3, 0].plot(t, N_H)
        ax_array[3, 0].set_title(r'$N_H$')
        #
        #
        ax_array[3, 1].plot(t, recovers)
        ax_array[3, 1].set_title(r'$R$')
        #
        ax_array[3, 2].plot(t, z)
        ax_array[3, 2].set_title(r'$z$')
        #
        for i in np.arange(3):
            ax_array[3, i].set(xlabel='time')
        for j in np.arange(4):
            ax_array[j, 0].set(ylabel='Individuals')
        #
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(self.plots_dir / 'populations_grid.png')
        plt.close(f1)
        #
        #
        plt.figure(2)
        #
        plt.subplot(211)
        plt.plot(t, z,
                 ls='-',
                 color='blue',
                 alpha=0.4
                 )
        plt.xlabel(r'time(weeks)')
        plt.ylabel(r'$p * (I_1 + I_2 + Y_{-1})$')
        #
        plt.subplot(212)
        plt.plot(t, Y_m1_h,
                 ls='-',
                 color='red',
                 alpha=0.4
                 )
        plt.xlabel(r'time(weeks)')
        plt.ylabel(r'$Y_{-1h}$')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.show()
        plt.savefig(self.plots_dir / 'DF_DHF.png')
        plt.close(2)

    def load_parameter_snapshot(self, file_name):
        """Load model parameters from a YAML file into the current instance."""
        with open(file_name, 'r') as f:
            parameter_data = yaml.safe_load(f)
        # Set initial conditions
        #
        self.Lambda_M = np.float64(parameter_data.get('Lambda_M'))
        self.beta_M = np.float64(parameter_data.get('beta_M'))
        self.beta_H = np.float64(parameter_data.get('beta_H'))
        self.b = np.float64(parameter_data.get('b'))
        self.mu_M = np.float64(parameter_data.get('mu_M'))
        self.alpha_c = np.float64(parameter_data.get('alpha_c'))
        self.alpha_h = np.float64(parameter_data.get('alpha_h'))
        self.sigma = np.float64(parameter_data.get('sigma'))
        self.p = np.float64(parameter_data.get('p'))
        self.theta = np.float64(parameter_data.get('theta'))
        self.M_s0 = np.float64(parameter_data.get('M_s0'))
        self.M_10 = np.float64(parameter_data.get('M_10'))
        self.M_20 = np.float64(parameter_data.get('M_20'))
        #
        #
        self.S_0 = np.float64(parameter_data.get('S_0'))
        self.I_10 = np.float64(parameter_data.get('I_10'))
        self.I_20 = np.float64(parameter_data.get('I_20'))
        self.S_m1_0 = np.float64(parameter_data.get('S_m1_0'))
        self.Y_m1c_0 = np.float64(parameter_data.get('Y_m1c_0'))
        self.Y_m1h_0 = np.float64(parameter_data.get('Y_m1h_0'))
        self.R_s0 = np.float64(parameter_data.get('R_s0'))
        self.R_s_m1_0 = np.float64(parameter_data.get('R_s_m1_0'))
        self.Rec_0 = np.float64(parameter_data.get('Rec_0'))
        self.z0 = np.float64(parameter_data.get('z0'))
        self.h = np.float64(parameter_data.get('h'))
        self.T = np.float64(parameter_data.get('T'))

    @classmethod
    def _flatten_parameter_payload(cls, payload: dict) -> dict:
        """Return a flat key/value parameter mapping from grouped or flat JSON."""
        if not isinstance(payload, dict):
            raise TypeError("Parameter payload must be a JSON object (dictionary).")

        flattened = {}
        for key, value in payload.items():
            if key in cls.PARAMETER_GROUP_KEYS and isinstance(value, dict):
                flattened.update(value)
                continue
            flattened[key] = value
        return flattened

    def load_parameters_from_json(self, file_path, strict=False):
        """Load model parameters from a JSON file and refresh derived fields."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_params = json.load(f)
        params = self._flatten_parameter_payload(raw_params)

        if strict:
            required_keys = {
                "Lambda_M",
                "Lambda_S",
                "Lambda_S_m1",
                "beta_M",
                "beta_H",
                "b",
                "mu_M",
                "mu_H",
                "alpha_c",
                "alpha_h",
                "sigma",
                "p",
                "theta",
                "M_s0",
                "M_10",
                "M_20",
                "S_0",
                "I_10",
                "I_20",
                "S_m1_0",
                "Y_m1_c0",
                "Y_m1_h0",
                "R_s0",
                "R_s_m1_0",
                "Rec_0",
                "z0",
                "N_S",
                "N_S_m1",
                "t0",
                "T",
                "grid_size",
                "h",
                "peak_df_cases",
                "r_01",
                "r_02",
                "r_zero",
            }
            missing_keys = sorted(key for key in required_keys if key not in params)
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise KeyError(
                    f"Missing required model parameters in {file_path}: {missing}."
                )

        fallback_defaults = {
            "Lambda_M": 60910.149999999994,
            "Lambda_S": 76.89246,
            "Lambda_S_m1": 1.13762149148,
            "beta_M": 0.01503,
            "beta_H": 0.95,
            "b": 21.875,
            "mu_M": 0.5075,
            "mu_H": 0.000273937,
            "alpha_c": 1.1655,
            "alpha_h": 1.1655,
            "sigma": 2.5,
            "p": 0.025,
            "theta": 0.05,
            "peak_df_cases": 550.0,
            "M_s0": 120000.0,
            "M_10": 10.0,
            "M_20": 10.0,
            "I_10": 120.0,
            "I_20": 40.0,
            "S_0": 278931.0,
            "S_m1_0": 4152.0,
            "Y_m1_c0": 0.0,
            "Y_m1_h0": 1.0,
            "R_s0": 0.0,
            "R_s_m1_0": 0.0,
            "Rec_0": 0.0,
            "z0": 4.00,
            "N_S": 280694.0,
            "N_S_m1": 4153.0,
            "N_M":120020,
            "t0": 25.0,
            "T": 53.0,
            "grid_size": 280000,
            "h": 0.00018928571428571428,
            "r_01": 0.0,
            "r_02": 0.0,
            "r_zero": 0.0,
        }

        def fallback_for(key):
            return getattr(self, key, fallback_defaults[key])

        def get_float(key):
            return np.float64(params.get(key, fallback_for(key)))

        def get_int(key):
            return int(params.get(key, fallback_for(key)))

        # Core rates
        self.Lambda_M = get_float("Lambda_M")
        self.beta_M = get_float("beta_M")
        self.beta_H = get_float("beta_H")
        self.b = get_float("b")
        self.mu_M = get_float("mu_M")
        self.mu_H = get_float("mu_H")
        self.alpha_c = get_float("alpha_c")
        self.alpha_h = get_float("alpha_h")
        self.sigma = get_float("sigma")
        self.p = get_float("p")
        self.theta = get_float("theta")
        self.peak_df_cases = get_float("peak_df_cases")

        # Vector initial conditions
        self.M_s0 = get_float("M_s0")
        self.M_10 = get_float("M_10")
        self.M_20 = get_float("M_20")

        # Host initial conditions
        self.I_10 = get_float("I_10")
        self.I_20 = get_float("I_20")
        self.S_0 = get_float("S_0")
        self.S_m1_0 = get_float("S_m1_0")
        self.Y_m1_c0 = get_float("Y_m1_c0")
        self.Y_m1_h0 = get_float("Y_m1_h0")
        self.R_s0 = get_float("R_s0")
        self.R_s_m1_0 = get_float("R_s_m1_0")
        self.Rec_0 = get_float("Rec_0")
        self.z0 = get_float("z0")

        # Timing / integration settings
        self.t0 = get_float("t0")
        self.T = get_float("T")
        self.grid_size = get_int("grid_size")
        self.h = get_float("h")
        self.r_01 = get_float("r_01")
        self.r_02 = get_float("r_02")
        self.r_zero = get_float("r_zero")

        # Derived populations and inflows
        self.N_S = get_float("N_S")
        self.N_S_m1 = get_float("N_S_m1")
        # Backward-compatible aliases retained for legacy call sites.
        self.N_H = self.N_S
        self.N_sm1 = self.N_S_m1
        self.Lambda_S_m1 = get_float("Lambda_S_m1")
        self.Lambda_S = get_float("Lambda_S")

        # Refresh integration arrays to match updated grid
        self.t = np.linspace(self.t0, self.T, self.grid_size)
        self.solution = self._build_solution_frame(
            self.t,
            np.zeros((len(self.t), len(self.ODE_STATE_COLUMNS)), dtype=np.float64),
        )
