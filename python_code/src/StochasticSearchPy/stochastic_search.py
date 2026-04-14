"""Core stochastic search model and plotting utilities."""

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
    """Run the two-strain dengue search model against prepared input data.

    Parameters
    ----------
    data_dir:
        Directory containing prepared input files.
    runtime_dir:
        Directory where plots, parameter snapshots, and other generated
        artifacts should be written.
    """

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
        self.df_error_threshold = 100
        self.dhf_error_threshold = 30
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
        self.Rec_0 = 0.0
        self.z0 = 1.050000
        self.N_H = self.S_0 + self.I_10 + self.I_20 + self.S_m1_0
        self.N_sm1 = self.S_m1_0 + self.Y_m1_c0 + self.Y_m1_h0
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
        self.solution = np.zeros([len(self.t), 13])
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
        sampled_weeks = np.round(time_grid[:-1:sample_stride]).astype(int)
        sampled_values = solution_values[:-1:sample_stride]
        return (
            np.delete(sampled_weeks, trim_indices),
            np.delete(sampled_values, trim_indices),
        )

    def save_fitting_plot(self):
        """Write the DF/DHF fitting comparison figure to ``runtime_dir``."""
        time_grid = self.t
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
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

        figure, axes = plt.subplots(2, 2, sharex=True)

        def calculate_padded_limits(*arrays, pad=0.1, floor=None):
            stacked = np.concatenate([np.asarray(a).ravel() for a in arrays if len(a) > 0])
            data_range = stacked.max() - stacked.min()
            pad_val = data_range * pad if data_range > 0 else pad
            lower = stacked.min() - pad_val
            if floor is not None:
                lower = max(floor, lower)
            return lower, stacked.max() + pad_val

        # Use dynamic y-limits so raw_data and simulation are both visible.
        df_ymin, df_ymax = calculate_padded_limits(weekly_df_counts, sampled_df_counts, pad=0.15, floor=0.0)
        dhf_ymin, dhf_ymax = calculate_padded_limits(weekly_dhf_counts, sampled_dhf_counts, pad=0.15, floor=0.0)
        df_xmin, df_xmax = calculate_padded_limits(df_observed_weeks, sampled_df_weeks, pad=0.02)
        dhf_xmin, dhf_xmax = calculate_padded_limits(dhf_observed_weeks, sampled_dhf_weeks, pad=0.02)

        axes[0, 0].plot(time_grid, z, 'b-')
        axes[0, 0].set_title(r'Reported DF ')
        axes[0, 0].set_xlim(df_xmin, df_xmax)
        axes[0, 0].set_ylim(df_ymin, df_ymax)

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
        axes[0, 1].text(27, 300,
                        'err=' + str(np.round(self.df_fit_error, 1)),
                        fontsize=10
                        )
        axes[0, 1].set_ylim(df_ymin, df_ymax)
        axes[0, 1].set_xlim(df_xmin, df_xmax)
        axes[0, 1].set_title(r'DF Fitting ')

        axes[1, 0].plot(time_grid, Y_m1_h, 'r-')
        axes[1, 0].set_title(r'DHF')
        axes[1, 0].set_xlim(dhf_xmin, dhf_xmax)
        axes[1, 0].set_ylim(dhf_ymin, dhf_ymax)
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
        axes[1, 1].text(27, 50,
                        'err=' + str(np.round(self.dhf_fit_error, 1)),
                        fontsize=10
                        )
        axes[1, 1].set_ylim(dhf_ymin, dhf_ymax)
        axes[1, 1].set_xlim(dhf_xmin, dhf_xmax)
        axes[1, 1].set_title(r'DHF Fitting ')

        for column_index in np.arange(2):
            axes[1, column_index].set(xlabel='week n')
        for row_index in np.arange(2):
            axes[row_index, 0].set(ylabel='Individuals')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(self.plots_dir / 'fitting_DF_DHF.png')
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
            # Tighten y-limits to keep dots visible around 1.
            axis.set_ylim(0.0, weekly_counts['count'].max() * 1.1)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'input_cases_timeseries.png')
        plt.close(fig)

    def compute_fitting_errors(self):
        """Compute the current DF and DHF fitting errors from the ODE solution."""
        time_grid = self.t
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
        self.peak_df_cases = np.max(z)
        fitting_window_weeks = 12
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
            q=None,
            **params,
    ):
        """Evaluate the ODE right-hand side from equations (3.1) to (3.3).

        The extra ``z`` state kept by the solver is interpreted as reported
        classical-incidence. That incidence term is inferred from the
        cumulative-incidence equations introduced immediately after (3.3), so
        the public solver interface remains unchanged.
        """
        del q  # Retained only for backward compatibility with older call sites.

        (
            M_s, M_I1, M_I2,
            S, I_1, I_2, R_s,
            S_m1, Y_m1_c, Y_m1_h, R_s_m1
        ) = state_vector

        (
            Lambda_M, Lambda_S, Lambda_S_m1, beta_M, beta_H, b, mu_M, mu_H,
            alpha_c, alpha_h, sigma, p, theta, n_vector, n_host, n_host_m1
        ) = operator.itemgetter(
            "Lambda_M", "Lambda_S", "Lambda_S_m1", "beta_M", "beta_H", "b", "mu_M", "mu_H",
            "alpha_c", "alpha_h", "sigma", "p", "theta", "n_vector", "n_host", "n_host_m1"
        )(params)

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

        return np.array(
            [
                    dM_s, dM_I1, dM_I2,
                    dS, dI_1, dI_2, dR_s,
                    dS_m1, dY_m1_c, dY_m1_h, dR_s_m1
            ],
            dtype=np.float64,
        )
#
    def solve_ode_system(self):
        """Integrate the model ODE system over the configured time grid."""
        final_time = self.T
        initial_time = self.t0
        time_grid = np.linspace(initial_time, final_time, self.grid_size)
        initial_state = np.array(
            [self.M_s0, self.M_10, self.M_20,
             self.S_0, self.I_10, self.I_20,
             self.S_m1_0, self.Y_m1_c0, self.Y_m1_h0,
             self.z0, self.Rec_0])
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

        solution = integrate.odeint(self.compute_ode_rhs, initial_state, time_grid,
                                    args=(Lambda_M, Lambda_S, Lambda_S_m1,
                                          beta_M, beta_H, b, mu_M, mu_H, alpha_c,
                                          alpha_h, sigma, p, theta))
        self.solution = solution
        self.t = time_grid
        return solution

    def sample_model_parameters(self, flag_deterministic=False):
        """Sample a new parameter set and update the model state in place.

        Parameters
        ----------
        flag_deterministic:
            When ``True``, load a fixed baseline parameter set instead of
            drawing random values.
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
            z0 = 1.050000
            #
            #
            self.N_H = S_0 + I_10 + I_20 + S_m1_0
            Y_m1_c0 = 0.0
            Y_m1_h0 = 0.0
            Rec_0 = 0.0
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
        R_s0 = self.R_s0
        R_s_m1_0 = self.R_s_m1_0
        self.z0 = z0
        #
        self.h = h
        self.T = T
        #
        parameter_vector = np.array([
            Lambda_M, beta_M, beta_H, b, mu_M, alpha_c, alpha_h,
            sigma, p, theta, M_s0, M_10, M_20, S_0, I_10, I_20,
            S_m1_0, Y_m1_c0, Y_m1_h0,
            R_s0, R_s_m1_0, z0, h, T,
        ])
        return parameter_vector

    def save_parameter_snapshot(self, file_name_prefix=None):
        """Persist the current parameter state as a YAML snapshot."""

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
        R_s_m1_0 =  self.R_s_m1_0
        z0 = self.z0
        h = self.h
        T = self.T
        r_zero = self.r_zero
        parameters = {
            'Lambda_M': Lambda_M, 'Lambda_S': Lambda_S,
            'Lambda_S_m1': Lambda_S_m1,
            'beta_M': beta_M, 'beta_H': beta_H,
            'b': b,
            'mu_M': mu_M, 'mu_H': mu_H,
            'alpha_c': alpha_c,
            'alpha_h': alpha_h,
            'sigma': sigma, 'p': p,
            'theta': theta,
            'M_s0': M_s0, 'M_10': M_10, 'M_20': M_20,
            'S_0': S_0, 'I_10': I_10, 'I_20': I_20,
            'S_m1_0': S_m1_0, 'Y_m1_c0': Y_m1_c0,
            'Y_m1_h0': Y_m1_h0,
            'R_s0': R_s0, 'R_s_m1_0': R_s_m1_0,
            'z0': z0, 'h': h,
            'T': T, 'r_zero': r_zero
            }
        #
        #
        #
        #
        str_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        prefix = file_name_prefix or str(self.output_dir / 'parameters_')
        file_name = prefix + str_time + '.yml'
        with open(file_name, 'w') as outfile:
            yaml.safe_dump(parameters, outfile, default_flow_style=False)

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
        # p = self.p
        # theta = self.theta
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

        M_s = self.solution[:, 0]
        M_1 = self.solution[:, 1]
        M_2 = self.solution[:, 2]
        S = self.solution[:, 3]
        I_1 = self.solution[:, 4]
        I_2 = self.solution[:, 5]
        S_m1 = self.solution[:, 6]
        Y_m1_c = self.solution[:, 7]
        Y_m1_h = self.solution[:, 8]
        z = self.solution[:, 9]
        recovers = self.solution[:, 10]
        #
        t = self.t
        N_H = S + I_1 + I_2 + S_m1 + Y_m1_c + Y_m1_h + recovers
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
