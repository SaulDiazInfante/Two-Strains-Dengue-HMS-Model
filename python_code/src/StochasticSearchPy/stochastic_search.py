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
except ImportError:
    # Fallback for running as a script (python stochastic_searchPy.py)
    import data_processing


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
    :ivar N_H: Total human population size.
    :type N_H: float
    :ivar N_sm1: Total susceptible and pre-symptomatic mosquito population size.
    :type N_sm1: float
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
    ODE_SOLUTION_COLUMNS = (TIME_GRID_COLUMN, *ODE_STATE_COLUMNS)
    FITTING_PLOT_FIGSIZE = (13.5, 7.5)
    FITTING_PLOT_LAYOUT = {
        "left": 0.075,
        "right": 0.985,
        "bottom": 0.095,
        "top": 0.925,
        "wspace": 0.18,
        "hspace": 0.34,
    }

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
        # Numerics initial parameters
        self.Lambda_M = 41933.0 * 7.0
        self.beta_M = 0.001372700
        self.beta_H = 0.042837955
        self.b = 3.807142 * 7.0
        self.mu_M = 0.07521661385714286 * 7.0
        self.mu_H = 0.000039 * 7.0
        self.alpha_c = 0.059245826 * 7.0
        self.alpha_h = 0.132552790 * 7.0
        self.sigma = 0.42264415014285717 * 7.0
        self.p = 0.050000
        self.theta = 0.086764968
        self.peak_df_cases = 1000
        #
        #
        self.M_s0 = 120000.000000
        self.M_10 = 20.000000
        self.M_20 = 30.000000
        #
        #
        self.I_10 = 10.000000
        self.I_20 = 20.000000
        self.S_0 = 35600.000000 - (self.I_10 + self.I_20)

        self.S_m1_0 = 4400.000000
        self.Y_m1_c0 = 0.0
        self.Y_m1_h0 = 0.0
        self.R_s0 = 0.0
        self.R_s_m1_0 = 0.0
        self.Rec_0 = 0.0

        self.z0 = 1.050000
        self.N_H = self.S_0 + self.I_10 + self.I_20 + self.R_s0
        self.N_sm1 = self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0 + self.R_s_m1_0
        self.Lambda_S_m1 = 0.1 * self.mu_H * self.N_H
        self.Lambda_S = 0.9 * self.mu_H * self.N_H
        self.t0 = 25  # 25.0
        self.T = 53
        self.grid_size = int(self.T - self.t0) * 10000
        self.h = np.float64(self.T) / np.float64(self.grid_size)
        self.r_01 = 0.0
        self.r_02 = 0.0
        self.r_zero = 0
        #
        self.t = np.linspace(self.t0, self.T, self.grid_size)
        self.solution = self._build_solution_frame(
            self.t,
            np.zeros((len(self.t), len(self.ODE_STATE_COLUMNS)), dtype=np.float64),
        )
        #
        self.df_fit_error = 0.0
        self.dhf_fit_error = 0.0
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
        meets_r_zero_threshold = self.r_zero > 1
        meets_df_error_threshold = self.df_fit_error < self.df_error_threshold
        meets_dhf_error_threshold = self.dhf_fit_error < self.dhf_error_threshold
        meets_peak_df_threshold = self.peak_df_cases < 700
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
            A 2x2 subplot figure comparing observed and simulated DF/DHF data.
        
        Notes
        -----
        The figure contains four subplots:
        - Top left: Reported DF time series
        - Top right: DF fitting comparison with error metric
        - Bottom left: DHF time series
        - Bottom right: DHF fitting comparison with error metric
        """
        solution_frame = self.solution
        time_grid = solution_frame[self.TIME_GRID_COLUMN].to_numpy()
        Y_m1_h = solution_frame["Y_m1_h"].to_numpy()
        z = solution_frame["z"].to_numpy()
        df_observed_weeks = self.weekly_df_frequency_array[3:, 0]
        dhf_observed_weeks = self.weekly_dhf_frequency_array[1:, 0]
        weekly_df_counts = self.weekly_df_frequency_array[3:, 1]
        weekly_dhf_counts = self.weekly_dhf_frequency_array[1:, 1]
        sample_stride = 10000
        df_trim_indices = [0, 1, 6, 9]
        dhf_trim_indices = [0, 1, 3, 4, 6, 7, 8]
        sampled_df_weeks, sampled_df_counts = self._sample_weekly_solution_points(
            time_grid, z, sample_stride, df_trim_indices
        )
        sampled_dhf_weeks, sampled_dhf_counts = self._sample_weekly_solution_points(
            time_grid, Y_m1_h, sample_stride, dhf_trim_indices
        )

        if figure is None:
            figure, axes = plt.subplots(
                2,
                2,
                sharex="row",
                figsize=self.FITTING_PLOT_FIGSIZE,
                constrained_layout=False,
            )
        else:
            figure.set_size_inches(*self.FITTING_PLOT_FIGSIZE, forward=True)
            if len(figure.axes) == 4:
                axes = np.asarray(figure.axes, dtype=object).reshape(2, 2)
                for axis in axes.flat:
                    axis.cla()
            else:
                figure.clear()
                axes = figure.subplots(2, 2, sharex="row")
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

        # Right column: fitting view scaled only by the observed frequencies.
        df_right_ymin, df_right_ymax = self._calculate_padded_limits(
            weekly_df_counts,
            pad=0.15,
            floor=0.0,
        )
        dhf_right_ymin, dhf_right_ymax = self._calculate_padded_limits(
            weekly_dhf_counts,
            pad=0.15,
            floor=0.0,
        )
        df_xmin, df_xmax = self._calculate_padded_limits(
            time_grid,
            df_observed_weeks,
            sampled_df_weeks,
            pad=0.02,
        )
        dhf_xmin, dhf_xmax = self._calculate_padded_limits(
            time_grid,
            dhf_observed_weeks,
            sampled_dhf_weeks,
            pad=0.02,
        )

        axes[0, 0].plot(time_grid, z, 'b-')
        axes[0, 0].set_title(r'Reported DF ')
        axes[0, 0].set_xlim(df_xmin, df_xmax)
        axes[0, 0].set_ylim(df_left_ymin, df_left_ymax)

        axes[0, 1].plot(df_observed_weeks, weekly_df_counts,
                        ls='--',
                        color='lightblue',
                        marker='o',
                        ms=8,
                        mfc='lightblue',
                        alpha=0.7)
        axes[0, 1].plot(time_grid, z,
                        ls=':',
                        color='darkblue',
                        alpha=0.3)
        axes[0, 1].plot(sampled_df_weeks, sampled_df_counts,
                        ls='none',
                        color='blue',
                        marker='*',
                        ms=8,
                        mfc='blue',
                        alpha=0.5)
        self._add_metric_annotation(
            axes[0, 1],
            'err=' + str(np.round(self.df_fit_error, 1)),
            (df_xmin, df_xmax),
            (df_right_ymin, df_right_ymax),
        )
        axes[0, 1].set_ylim(df_right_ymin, df_right_ymax)
        axes[0, 1].set_xlim(df_xmin, df_xmax)
        axes[0, 1].set_title(r'DF Fitting ')

        axes[1, 0].plot(time_grid, Y_m1_h, 'r-')
        axes[1, 0].set_title(r'DHF')
        axes[1, 0].set_xlim(dhf_xmin, dhf_xmax)
        axes[1, 0].set_ylim(dhf_left_ymin, dhf_left_ymax)
        axes[1, 1].plot(dhf_observed_weeks, weekly_dhf_counts,
                        ls='--',
                        color='orange',
                        marker='o',
                        ms=8,
                        mfc='orange',
                        alpha=0.5)
        axes[1, 1].plot(time_grid, Y_m1_h,
                        ls=':',
                        color='crimson',
                        alpha=0.5
                        )
        axes[1, 1].plot(sampled_dhf_weeks, sampled_dhf_counts,
                        ls='none',
                        color='crimson',
                        marker='*'
                        )
        self._add_metric_annotation(
            axes[1, 1],
            'err=' + str(np.round(self.dhf_fit_error, 1)),
            (dhf_xmin, dhf_xmax),
            (dhf_right_ymin, dhf_right_ymax),
        )
        axes[1, 1].set_ylim(dhf_right_ymin, dhf_right_ymax)
        axes[1, 1].set_xlim(dhf_xmin, dhf_xmax)
        axes[1, 1].set_title(r'DHF Fitting ')

        for column_index in np.arange(2):
            axes[1, column_index].set(xlabel='week n')
            axes[1, column_index].xaxis.set_label_coords(0.5, -0.14)
        for row_index in np.arange(2):
            axes[row_index, 0].set(ylabel='Individuals')
            axes[row_index, 0].yaxis.set_label_coords(-0.075, 0.5)

        return figure
    
    def save_fitting_plot(self):
        """Save the DF/DHF fitting comparison figure to disk.
    
        Creates a 2x2 subplot figure comparing observed weekly DF/DHF counts
        with model simulation results and saves it to the plots directory.
        The file is saved as ``fitting_DF_DHF.png`` in ``self.plots_dir``.
        """
        figure = self.create_fitting_plot()
        figure.savefig(self.plots_dir / 'fitting_DF_DHF.png')
        plt.close(figure)
    def save_input_data_plot(self):
        """Plot daily DF and DHF case time series from CSVs in two stacked axes.

        Reads ``incidence_data_DF.csv`` and ``incidence_data_DHF.csv`` from
        ``self.data_dir`` with the first column parsed as dates. Produces a
        2-row by 1-column figure saved to ``plots/input_cases_timeseries.png``.
        """
        df_incidence_path = self.data_dir / 'incidence_data_DF.csv'
        dhf_incidence_path = self.data_dir / 'incidence_data_DHF.csv'

        # Expect two columns: date, label. Parse the first as datetime.
        # Date strings are in M/D/YYYY format. The DHF file already contains a header.
        date_format = '%m/%d/%Y'
        df_case_rows = pd.read_csv(
            df_incidence_path,
            header=0,
            names=['date', 'incidence'],
            parse_dates=['date'],
            date_format=date_format,
        )
        # Each row is a single reported case; set numeric incidence count.
        dhf_case_rows = pd.read_csv(
            dhf_incidence_path,
            header=0,
            names=['date', 'incidence'],
            parse_dates=['date'],
            date_format=date_format,
        )
        def summarize_cases_by_week_start(incidence_rows: pd.DataFrame) -> pd.DataFrame:
            # Convert each record's date to the start of its ISO week to retain a
            # calendar-aware type (Timestamp) instead of a plain integer week
            # number. This keeps downstream CSVs and plots date-typed weeks.
            week_starts = incidence_rows['date'].dt.to_period('W').dt.start_time
            weekly_counts = incidence_rows.groupby(week_starts).size().reset_index(name='count')
            weekly_counts.rename(columns={weekly_counts.columns[0]: 'week'}, inplace=True)
            return weekly_counts

        weekly_df_counts = summarize_cases_by_week_start(df_case_rows)
        weekly_dhf_counts = summarize_cases_by_week_start(dhf_case_rows)
        # Ensure week column is explicitly datetime-typed (start of week).
        for weekly_counts in (weekly_df_counts, weekly_dhf_counts):
            weekly_counts['week'] = pd.to_datetime(weekly_counts['week'])
        weekly_df_counts.to_csv(self.build_data_file_path('frequency_per_week_DF.csv'),
                                index=False)
        weekly_dhf_counts.to_csv(
            self.build_data_file_path('frequency_per_week_DHF.csv'),
            index=False
        )

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
        """Compute the current DF and DHF fitting errors from the ODE solution."""
        solution_frame = self.solution
        time_grid = solution_frame[self.TIME_GRID_COLUMN].to_numpy()
        Y_m1_h = solution_frame["Y_m1_h"].to_numpy()
        z = solution_frame["z"].to_numpy()
        self.peak_df_cases = np.max(z)
        fitting_window_weeks = 10
        sample_stride = 10000
        df_trim_indices = [0, 1, 6, 9]
        dhf_trim_indices = [0, 1, 3, 4, 6, 7, 8]
        weekly_df_counts = self.weekly_df_frequency_array[3:, 1]
        weekly_dhf_counts = self.weekly_dhf_frequency_array[1:, 1]
        _, sampled_df_counts = self._sample_weekly_solution_points(
            time_grid, z, sample_stride, df_trim_indices
        )
        _, sampled_dhf_counts = self._sample_weekly_solution_points(
            time_grid, Y_m1_h, sample_stride, dhf_trim_indices
        )
        df_fit_error = np.linalg.norm(
            weekly_df_counts[:fitting_window_weeks] - sampled_df_counts[:fitting_window_weeks],
            ord=np.inf,
        )
        self.df_fit_error = df_fit_error
        dhf_fit_error = np.linalg.norm(
            weekly_dhf_counts[:fitting_window_weeks] - sampled_dhf_counts[:fitting_window_weeks],
            ord=np.inf,
        )
        self.dhf_fit_error = dhf_fit_error

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
        if state_vector.shape[0] == len(StochasticSearch.ODE_STATE_COLUMNS):
            (
                M_s, M_I1, M_I2,
                S, I_1, I_2, R_s,
                S_m1, Y_m1_c, Y_m1_h, R_s_m1,
                z,
            ) = state_vector
            include_z_state = True
        elif state_vector.shape[0] == len(StochasticSearch.ODE_STATE_COLUMNS) - 1:
            (
                M_s, M_I1, M_I2,
                S, I_1, I_2, R_s,
                S_m1, Y_m1_c, Y_m1_h, R_s_m1,
            ) = state_vector
            z = np.float64(0.0)
            include_z_state = False
        else:
            raise ValueError(
                "state_vector must contain either 11 legacy states or 12 solver states; "
                f"received shape {state_vector.shape}."
            )

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
        dR_s = alpha_c * (I_1 + I_2) - mu_H * R_s

        dS_m1 = Lambda_S_m1 - reinfection_incidence - mu_H * S_m1
        dY_m1_c = (1.0 - theta) * reinfection_incidence - (alpha_c + mu_H) * Y_m1_c
        dY_m1_h = theta * reinfection_incidence - (alpha_h + mu_H) * Y_m1_h
        dR_s_m1 = alpha_c * Y_m1_c + alpha_h * Y_m1_h - mu_H * R_s_m1
        dz = p * (I_1 + I_2 + Y_m1_c)

        rhs = [
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
        ]
        if not include_z_state:
            rhs = rhs[:-1]
        rhs = np.array(rhs, dtype=np.float64)
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
        #
        #
        if flag_deterministic:
            Lambda_M = 41933.0 * 7.0
            beta_M = 0.001372700
            beta_H = 0.042837955
            b = 3.807142 * 7.0
            mu_M = 0.07521661385714286 * 7.0
            Lambda_S = 1.4 * 7.0  # Modify to get N_H constant
            mu_H = 0.000039 * 7.0
            alpha_c = 0.059245826 * 7.0
            alpha_h = 0.132552790 * 7.0
            Lambda_S_m1 = 0.156 * 7
            sigma = 0.42264415014285717 * 7.0
            p = 0.250000
            theta = 0.086764968
            #
            #
            M_s0 = 120000.000000
            M_10 = 20.000000
            M_20 = 30.000000
            S_0 = 35600.000000
            I_10 = 1.000000
            I_20 = 20.000000
            S_m1_0 = 4400.000000
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            R_s0 = 0.0
            R_s_m1_0 = 0.0
            Rec_0 = 0.0
            z0 = 1.050000
            z0 = p * (I_10 + I_20 + Y_m1_c0)
        else:
            Lambda_M = 7 * np.abs(6000 + 2000 * np.random.randn())
            beta_M = 0.05 * np.random.rand()
            b = np.abs(2 + np.random.randn()) * 7
            mu_M = (0.033 + 0.067 * np.random.rand()) * 7
            #
            # Human constants
            #
            beta_H = 0.05 * np.random.rand()
            # for recovering
            alpha_c = (0.0556 + .0444 * np.random.rand()) * 7
            alpha_h = (0.125 + 0.125 * np.random.rand()) * 7
            sigma = 0.5 + 4.5 * np.random.rand()
            #
            p = 0.1 + .2 * np.random.rand()
            # p = .05
            theta = 0.01 + 0.5 * np.random.rand()
            #
            # Retain these legacy draws so the stochastic sampling stream stays
            # unchanged, even though the current mosquito initial conditions are fixed.
            _legacy_primary_mosquito_share = 0.8 + .1 * np.random.rand()
            _legacy_infected_mosquito_draws = np.random.rand(2)
            _legacy_infected_mosquito_weights = (
                0.8
                * (1.0 - _legacy_primary_mosquito_share)
                / _legacy_infected_mosquito_draws.sum()
                * _legacy_infected_mosquito_draws
            )

            M_s0 = 120000
            M_10 = 10
            M_20 = 10
            #
            # Initial condition humans
            #
            N_H = 37000 + 5000 * np.random.rand()
            N_S = .9 * N_H
            self.N_H = N_H
            mu_H = self.mu_H
            Lambda_S = mu_H * N_S
            Lambda_S_m1 = mu_H * (N_H - N_S)
            # partition

            I_10 = 1.000000
            I_20 = 1.000000
            S_0 = 35600.000000 - (I_10 + I_20)
#
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            S_m1_0 = 4400.000000 - (Y_m1_c0 + Y_m1_h0)
            N_H = S_0 + I_10 + I_20 + S_m1_0
            #
            #
            #
            #
            R_s0 = 0.0
            R_s_m1_0 = 0.0
            Rec_0 = 0.0
            z0 = p * (I_10 + I_20 + Y_m1_c0)
        #
        #
        # Numerical parameters
        #
        T = self.T
        h = np.float64(self.T) / np.float64(self.grid_size)
        #
        # Object parameters update
        #
        self.Lambda_M = Lambda_M
        self.Lambda_S = Lambda_S
        self.Lambda_S_m1 = Lambda_S_m1
        self.beta_M = beta_M
        self.beta_H = beta_H
        self.b = b
        self.mu_M = mu_M
        self.mu_H = mu_H
        self.alpha_c = alpha_c
        self.alpha_h = alpha_h
        self.sigma = sigma
        self.p = p
        self.theta = theta
        #
        self.M_s0 = M_s0
        self.M_10 = M_10
        self.M_20 = M_20
        #
        self.S_0 = S_0
        self.I_10 = I_10
        self.I_20 = I_20
        #
        self.S_m1_0 = S_m1_0
        #
        self.Y_m1_c0 = Y_m1_c0
        self.Y_m1_h0 = Y_m1_h0
        self.R_s0 = R_s0
        self.R_s_m1_0 = R_s_m1_0
        self.Rec_0 = Rec_0
        self.z0 = z0
        #
        self.N_H = self.S_0 + self.I_10 + self.I_20 + self.R_s0
        self.N_sm1 = self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0 + self.R_s_m1_0
        self.h = h
        self.T = T
        sampled_parameter_frame = self._build_sample_parameter_frame()
        return sampled_parameter_frame

    def save_parameter_snapshot(self, file_name_prefix=None):
        """Persist the current parameter state as a JSON snapshot."""

        # load parameters
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
        M_s0 = self.M_s0
        M_10 = self.M_10
        M_20 = self.M_20
        S_0 = self.S_0
        I_10 = self.I_10
        I_20 = self.I_20
        S_m1_0 = self.S_m1_0
        Y_m1_c0 = self.Y_m1_c0
        Y_m1_h0 = self.Y_m1_h0
        R_s0 = self.R_s0
        R_s_m1_0 = self.R_s_m1_0
        z0 = self.z0
        h = self.h
        T = self.T
        r_zero = self.r_zero
        parameters = {
            'Lambda_M': float(Lambda_M),
            'Lambda_S': float(Lambda_S),
            'Lambda_S_m1': float(Lambda_S_m1),
            'beta_M': float(beta_M),
            'beta_H': float(beta_H),
            'b': float(b),
            'mu_M': float(mu_M),
            'mu_H': float(mu_H),
            'alpha_c': float(alpha_c),
            'alpha_h': float(alpha_h),
            'sigma': float(sigma),
            'p': float(p),
            'theta': float(theta),
            'M_s0': float(M_s0),
            'M_10': float(M_10),
            'M_20': float(M_20),
            'S_0': float(S_0),
            'I_10': float(I_10),
            'I_20': float(I_20),
            'S_m1_0': float(S_m1_0),
            'Y_m1_c0': float(Y_m1_c0),
            'Y_m1_h0': float(Y_m1_h0),
            'R_s0': float(R_s0),
            'R_s_m1_0': float(R_s_m1_0),
            'z0': float(z0),
            'h': float(h),
            'T': float(T),
            'r_zero': float(r_zero)
        }

        str_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        prefix = file_name_prefix or str(self.output_dir / 'parameters_')
        file_name = prefix + str_time + '.json'
        with open(file_name, 'w') as outfile:
            json.dump(parameters, outfile, indent=2)

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
        N_H = self.N_H
        N_sm1 = self.N_sm1
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

    def load_parameters_from_json(self, file_path):
        """Load model parameters from a JSON file and refresh derived fields."""
        with open(file_path, 'r') as f:
            params = json.load(f)

        def get_float(key, default):
            return np.float64(params.get(key, default))

        def get_int(key, default):
            return int(params.get(key, default))

        # Core rates
        self.Lambda_M = get_float('Lambda_M', self.Lambda_M)
        self.beta_M = get_float('beta_M', self.beta_M)
        self.beta_H = get_float('beta_H', self.beta_H)
        self.b = get_float('b', self.b)
        self.mu_M = get_float('mu_M', self.mu_M)
        self.mu_H = get_float('mu_H', self.mu_H)
        self.alpha_c = get_float('alpha_c', self.alpha_c)
        self.alpha_h = get_float('alpha_h', self.alpha_h)
        self.sigma = get_float('sigma', self.sigma)
        self.p = get_float('p', self.p)
        self.theta = get_float('theta', self.theta)
        self.peak_df_cases = get_float('peak_df_cases', self.peak_df_cases)

        # Vector initial conditions
        self.M_s0 = get_float('M_s0', self.M_s0)
        self.M_10 = get_float('M_10', self.M_10)
        self.M_20 = get_float('M_20', self.M_20)

        # Host initial conditions
        self.I_10 = get_float('I_10', self.I_10)
        self.I_20 = get_float('I_20', self.I_20)
        self.S_0 = get_float('S_0', self.S_0)
        self.S_m1_0 = get_float('S_m1_0', self.S_m1_0)
        self.Y_m1_c0 = get_float('Y_m1_c0', self.Y_m1_c0)
        self.Y_m1_h0 = get_float('Y_m1_h0', self.Y_m1_h0)
        self.Rec_0 = get_float('Rec_0', self.Rec_0)
        self.z0 = get_float('z0', self.z0)

        # Timing / integration settings
        self.t0 = get_float('t0', self.t0)
        self.T = get_float('T', self.T)
        self.grid_size = get_int('grid_size', int((self.T - self.t0) * 10000))
        self.h = get_float('h', np.float64(self.T) / np.float64(self.grid_size))
        self.r_01 = get_float('r_01', self.r_01)
        self.r_02 = get_float('r_02', self.r_02)
        self.r_zero = get_float('r_zero', self.r_zero)

        # Derived populations and inflows (use provided value if present, else recompute)
        self.N_H = get_float('N_H', self.S_0 + self.I_10 + self.I_20 + self.S_m1_0)
        self.N_sm1 = get_float('N_sm1', self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0)
        self.Lambda_S_m1 = get_float('Lambda_S_m1', 0.1 * self.mu_H * self.N_H)
        self.Lambda_S = get_float('Lambda_S', 0.9 * self.mu_H * self.N_H)

        # Refresh integration arrays to match updated grid
        self.t = np.linspace(self.t0, self.T, self.grid_size)
        self.solution = self._build_solution_frame(
            self.t,
            np.zeros((len(self.t), len(self.ODE_STATE_COLUMNS)), dtype=np.float64),
        )
