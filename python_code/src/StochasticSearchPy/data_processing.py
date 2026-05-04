"""Data preparation utilities for the dengue model package."""

import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import csv
import subprocess
from pathlib import Path
import pandas as pd

try:
    from .data_assets import resolve_data_directory
except ImportError:
    from data_assets import resolve_data_directory


class DataProcessing:
    """Load, normalize, and aggregate dengue incidence input data.

    Parameters
    ----------
    data_dir:
        Directory containing prepared model input files. When omitted, the
        package resolves the directory from the environment or a source
        checkout.
    """

    def __init__(self, data_dir=None):
        self.df_incidence_rows = []
        self.dhf_incidence_rows = []
        self.daily_df_frequency_table = pd.DataFrame()
        self.daily_dhf_frequency_table = pd.DataFrame()
        self.weekly_df_frequency_table = pd.DataFrame()
        self.weekly_dhf_frequency_table = pd.DataFrame()
        self.weekly_df_frequency_array = np.empty((0, 2), dtype=int)
        self.weekly_dhf_frequency_array = np.empty((0, 2), dtype=int)
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = resolve_data_directory(data_dir, create=data_dir is not None)
        setup_script = self.data_dir / "setup.sh"
        if setup_script.exists():
            subprocess.run(["sh", str(setup_script)], check=False)
        self.connection = sqlite3.connect(self.build_data_file_path('dengue_data_2010.sqlite'))
        self.cur = self.connection.cursor()

    def build_data_file_path(self, filename: str) -> str:
        """Return the absolute path for a file inside the configured data directory."""
        return str(self.data_dir / filename)

    def close_database_connection(self):
        """Close the underlying SQLite connection if it is open."""
        if getattr(self, "connection", None) is not None:
            self.connection.close()

    def __enter__(self):
        """Return the processor so it can be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close database resources when leaving a context manager block."""
        self.close_database_connection()

    @staticmethod
    def convert_excel_date_to_datetime(excel_date):
        """Convert an Excel day offset into a Python ``datetime``."""
        temp = datetime.datetime(1900, 1, 1)
        delta = datetime.timedelta(days=excel_date)
        return temp + delta

    def convert_excel_dates_to_datetimes(self, dates_excel):
        """Convert an iterable of Excel day offsets into Python datetimes."""
        return [self.convert_excel_date_to_datetime(day) for day in dates_excel]

    def check_table_has_rows(self, table_name: str) -> bool:
        with self.connection as conn:
            cur = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if cur.fetchone() is None:
                return False
            cur = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1;")
            return cur.fetchone() is not None

    def create_nominal_data_table(self):
        with self.connection as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nominal_data (
                    OBJECTID, Join_Count, TARGET_FID, ID_HILLO, ID_SON, ANYO,
                    FOLIO, SEXO, EDAD, D_H_, U_NOTIFICA, J_N_, DOMICILIO,
                    COLONIA, LOCALIDAD, MUNICIPIO, I_CUADRO, I_CUADRO2,
                    S_E__INICI, J_S_, HOSP, HEMORRAGIA, PLAQUETAS, DEFUNCION,
                    FOLIO_DENG, DIAS_EVOLU, IgM_ELISA, IgG_ELISA, NS1,
                    AISLAMIENT, FD_FHD, DIAGNOSTIC, OBSERVACIO, ID_HILLO_1,
                    EDAD2, CUANTIFICA, GRAVE, CVEGEO, CVEGEO_1, POB1, POB26_R,
                    POB27_R, POB30_R, ECO25_R, SALUD2_R, EDU34_R, MIG11_R,
                    VIV2_R, VIV4_R, VIV9_R, VIV17_R, VIV26_R, VIV27_R, VIV28_R,
                    VIV31_R, VIV32_R, VIV33_R, VIV34_R, VIV35_R, VIV36_R,
                    GRADO_MARG, GRADO_MA_1, hectarea, DENSI_POB, VIV2,
                    INDEX_NORM, INDEX_NO_1, x, y
                );
                """
            )

    @staticmethod
    def find_matching_indices(lst, element_to_match):
        """Return all indices where ``element_to_match`` occurs in ``lst``."""
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element_to_match, offset + 1)
            except ValueError:
                return result
            result.append(offset)

    def populate_nominal_data_table(self):
        """Populate the ``nominal_data`` SQLite table from the reference CSV.

        If the table already exists and contains rows, it is reused as-is.
        """
        if self.check_table_has_rows("nominal_data"):
            return None

        # Fallback: build the table from the CSV if available.
        csv_path = Path(self.build_data_file_path('casos_dengue_2010_AGEB2010.csv'))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing raw_data file {csv_path}. Provide the CSV or an "
                "already-populated nominal_data table."
            )

        self.create_nominal_data_table()
        with open(csv_path, 'r', newline='') as file_in, self.connection as conn:
            to_db = [
                (
                    i["OBJECTID"], i["Join_Count"], i["TARGET_FID"],
                    i["ID_HILLO"], i["ID_SON"], i["ANYO"], i["FOLIO"],
                    i["SEXO"], i["EDAD"], i["D_H_"], i["U_NOTIFICA"],
                    i["J_N_"], i["DOMICILIO"], i["COLONIA"],
                    i["LOCALIDAD"],
                    i["MUNICIPIO"], i["I_CUADRO"], i["I_CUADRO2"],
                    i["S_E__INICI"], i["J_S_"], i["HOSP"],
                    i["HEMORRAGIA"],
                    i["PLAQUETAS"], i["DEFUNCION"], i["FOLIO_DENG"],
                    i["DIAS_EVOLU"], i["IgM_ELISA"], i["IgG_ELISA"],
                    i["NS1"],
                    i["AISLAMIENT"], i["FD_FHD"], i["DIAGNOSTIC"],
                    i["OBSERVACIO"], i["ID_HILLO_1"], i["EDAD2"],
                    i["CUANTIFICA"], i["GRAVE"], i["CVEGEO"],
                    i["CVEGEO_1"],
                    i["POB1"], i["POB26_R"], i["POB27_R"],
                    i["POB30_R"],
                    i["ECO25_R"], i["SALUD2_R"], i["EDU34_R"],
                    i["MIG11_R"],
                    i["VIV2_R"], i["VIV4_R"], i["VIV9_R"],
                    i["VIV17_R"],
                    i["VIV26_R"], i["VIV27_R"], i["VIV28_R"],
                    i["VIV31_R"],
                    i["VIV32_R"], i["VIV33_R"], i["VIV34_R"],
                    i["VIV35_R"],
                    i["VIV36_R"], i["GRADO_MARG"], i["GRADO_MA_1"],
                    i["hectarea"], i["DENSI_POB"], i["VIV2"],
                    i["INDEX_NORM"], i["INDEX_NO_1"], i["x"], i["y"]
                    )
                for i in csv.DictReader(file_in)
                ]
            conn.executemany(
                """
                INSERT INTO nominal_data (
                    OBJECTID, Join_Count, TARGET_FID, ID_HILLO, ID_SON, ANYO,
                    FOLIO, SEXO, EDAD, D_H_, U_NOTIFICA, J_N_, DOMICILIO,
                    COLONIA, LOCALIDAD, MUNICIPIO, I_CUADRO, I_CUADRO2,
                    S_E__INICI, J_S_, HOSP, HEMORRAGIA, PLAQUETAS, DEFUNCION,
                    FOLIO_DENG, DIAS_EVOLU, IgM_ELISA, IgG_ELISA, NS1,
                    AISLAMIENT, FD_FHD, DIAGNOSTIC, OBSERVACIO, ID_HILLO_1,
                    EDAD2, CUANTIFICA, GRAVE, CVEGEO, CVEGEO_1, POB1, POB26_R,
                    POB27_R, POB30_R, ECO25_R, SALUD2_R, EDU34_R, MIG11_R,
                    VIV2_R, VIV4_R, VIV9_R, VIV17_R, VIV26_R, VIV27_R, VIV28_R,
                    VIV31_R, VIV32_R, VIV33_R, VIV34_R, VIV35_R, VIV36_R,
                    GRADO_MARG, GRADO_MA_1, hectarea, DENSI_POB, VIV2,
                    INDEX_NORM, INDEX_NO_1, x, y
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
                """,
                to_db,
            )
        return None

#    @staticmethod
    def rebuild_incidence_data_table(self):
        """Create the normalized ``incidence_data`` table used by the model."""
        # Ensure we always have the expected schema
        self.cur.execute("DROP TABLE IF EXISTS incidence_data;")
        self.cur.execute("CREATE TABLE incidence_data \
                        (case_date, case_week, fever, x, y);")

        # Populate directly from nominal_data if available.
        self.cur.execute("SELECT name FROM sqlite_master "
                         "WHERE type='table' AND name='nominal_data';")
        if self.cur.fetchone():
            self.cur.execute("INSERT INTO incidence_data(case_date, case_week, fever, x, y) "
                             "SELECT I_CUADRO, S_E__INICI, FD_FHD, x, y "
                             "FROM nominal_data;")
        else:
            # As a last resort, fall back to the CSV (if the user provides it).
            with open(self.build_data_file_path('casos_dengue_2010_AGEB2010.csv'), 'r', newline='') as file_in:
                data_record_lines = csv.DictReader(file_in)
                to_db = [
                    (
                        i["I_CUADRO"], i["S_E__INICI"], i["FD_FHD"], i["x"], i["y"]
                        )
                    for i in data_record_lines
                    ]
            self.cur.executemany("INSERT INTO incidence_data(\
                        case_date, case_week, fever, x, y)\
                        VALUES (?, ?, ?, ?, ?);", to_db)
        self.connection.commit()
        return None

    def export_incidence_rows_with_weeks(self):
        """Export DF and DHF incidence rows including date and epidemiological week."""
        self.populate_nominal_data_table()
        self.rebuild_incidence_data_table()
        query_buffer_DF = "SELECT case_date, case_week, \
                            fever FROM incidence_data WHERE fever = 'FD'"
        self.cur.execute(query_buffer_DF)
        result_DF = self.cur.fetchall()
        with open(self.build_data_file_path('incidence_data_DF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DF)
        #
        # frequency per day counting
        #
        query_buffer_DHF = "SELECT case_date, case_week, \
                           fever FROM incidence_data WHERE fever = 'FHD'"
        self.cur.execute(query_buffer_DHF)
        result_DHF = self.cur.fetchall()
        with open(self.build_data_file_path('incidence_data_DHF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DHF)

        self.dhf_incidence_rows = result_DHF
        self.df_incidence_rows = result_DF
        return result_DF, result_DHF
    
    def export_incidence_rows(self):
        """Export DF and DHF incidence rows keyed only by case date."""
        self.populate_nominal_data_table()
        self.rebuild_incidence_data_table()
        query_buffer_DF = "SELECT case_date, \
                            fever FROM incidence_data WHERE fever = 'FD'"
        self.cur.execute(query_buffer_DF)
        result_DF = self.cur.fetchall()
        with open(self.build_data_file_path('incidence_data_DF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DF)
        #
        # frequency per day counting
        #
        query_buffer_DHF = "SELECT case_date, \
                           fever FROM incidence_data WHERE fever = 'FHD'"
        self.cur.execute(query_buffer_DHF)
        result_DHF = self.cur.fetchall()
        with open(self.build_data_file_path('incidence_data_DHF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DHF)

        self.dhf_incidence_rows = result_DHF
        self.df_incidence_rows = result_DF
        return result_DF, result_DHF

    def load_incidence_dataframes(self):
        """Load DF and DHF incidence CSVs, generating them from SQLite if needed."""
        df_path = Path(self.build_data_file_path("incidence_data_DF.csv"))
        dhf_path = Path(self.build_data_file_path("incidence_data_DHF.csv"))
        if not df_path.exists() or not dhf_path.exists():
            self.export_incidence_rows()

        columns = ["date", "Fever"]
        df_df = pd.read_csv(
            df_path,
            header=None,
            names=columns,
            parse_dates=["date"],
            dayfirst=False,
        )
        df_dhf = pd.read_csv(
            dhf_path,
            header=None,
            names=columns,
            parse_dates=["date"],
            dayfirst=False,
        )
        df_df.set_index("date", inplace=True)
        df_dhf.set_index("date", inplace=True)
        return df_df, df_dhf

    @staticmethod
    def count_cases_by_date(frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate case rows into a dense daily count table.

        Missing calendar dates between the first and last reported cases are
        included with a zero count so downstream rolling-window operations keep
        their intended day-based meaning.
        """
        if frame.empty:
            empty_counts = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            empty_counts["count"] = pd.Series(dtype=int)
            return empty_counts

        case_dates = pd.to_datetime(frame.index).normalize()
        counts = (
            pd.Series(1, index=case_dates, name="count")
            .groupby(level=0)
            .sum()
            .sort_index()
            .to_frame()
        )
        full_date_index = pd.date_range(
            start=counts.index.min(),
            end=counts.index.max(),
            freq="D",
            name="date",
        )
        counts = counts.reindex(full_date_index, fill_value=0)
        counts["count"] = counts["count"].astype(int)
        return counts

    @staticmethod
    def read_daily_frequency_table(path: Path) -> pd.DataFrame:
        """Load a persisted daily count CSV with explicit ``date`` and ``count`` columns."""
        frame = pd.read_csv(path)
        required_columns = {"date", "count"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                f"Expected columns {sorted(required_columns)} in {path}, "
                f"found {list(frame.columns)}."
            )

        counts = frame.loc[:, ["date", "count"]].copy()
        counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()
        counts["count"] = pd.to_numeric(counts["count"], errors="raise").astype(int)
        counts = counts.groupby("date", as_index=False)["count"].sum().sort_values("date")
        counts.set_index("date", inplace=True)
        full_date_index = pd.date_range(
            start=counts.index.min(),
            end=counts.index.max(),
            freq="D",
            name="date",
        )
        counts = counts.reindex(full_date_index, fill_value=0)
        counts.index.name = "date"
        return counts

    @staticmethod
    def compute_moving_average(
        values: pd.Series | pd.DataFrame,
        window_days: int,
    ) -> pd.Series:
        """Return a trailing moving average over daily counts."""
        if isinstance(values, pd.DataFrame):
            if "count" not in values.columns:
                raise KeyError("Daily count dataframe must contain a 'count' column.")
            series = values["count"]
        else:
            series = values
        return series.astype(float).rolling(
            window=window_days,
            min_periods=window_days,
        ).mean()

    @staticmethod
    def aggregate_daily_counts_by_week(frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a date-frequency table into one count per ISO week."""
        weekly = (
            frame.reset_index(drop=False)
            .assign(week=lambda df: df["date"].dt.isocalendar().week.astype(int))
            .groupby("week", as_index=False)["count"]
            .sum()
            .sort_values("week")
            .reset_index(drop=True)
        )
        weekly["count"] = weekly["count"].astype(int)
        return weekly

    def build_daily_frequency_tables(self, persist: bool = True):
        """Return DF and DHF daily frequency tables and optionally persist them as CSV."""
        df_df, df_dhf = self.load_incidence_dataframes()
        freq_df = self.count_cases_by_date(df_df)
        freq_dhf = self.count_cases_by_date(df_dhf)
        freq_df["count"] = freq_df["count"].astype(int)
        freq_dhf["count"] = freq_dhf["count"].astype(int)
        self.daily_df_frequency_table = freq_df
        self.daily_dhf_frequency_table = freq_dhf

        if persist:
            freq_df.reset_index().to_csv(
                self.build_data_file_path("frequency_per_date_DF.csv"),
                index=False,
                date_format="%Y-%m-%d",
            )
            freq_dhf.reset_index().to_csv(
                self.build_data_file_path("frequency_per_date_DHF.csv"),
                index=False,
                date_format="%Y-%m-%d",
            )
        return freq_df, freq_dhf

    def build_weekly_frequency_tables(self):
        """Return DF and DHF weekly frequency tables and persist them as CSV."""
        freq_date_df, freq_date_dhf = self.build_daily_frequency_tables()
        freq_df = self.aggregate_daily_counts_by_week(freq_date_df)
        freq_dhf = self.aggregate_daily_counts_by_week(freq_date_dhf)

        self.weekly_df_frequency_table = freq_df
        self.weekly_dhf_frequency_table = freq_dhf
        weekly_df_array = freq_df[["week", "count"]].to_numpy(dtype=int)
        weekly_dhf_array = freq_dhf[["week", "count"]].to_numpy(dtype=int)
        self.weekly_df_frequency_array = weekly_df_array
        self.weekly_dhf_frequency_array = weekly_dhf_array

        freq_df.to_csv(self.build_data_file_path("frequency_per_week_DF.csv"), index=False)
        freq_dhf.to_csv(self.build_data_file_path("frequency_per_week_DHF.csv"), index=False)
        return freq_df, freq_dhf

    def build_weekly_frequency_arrays(self):
        """Return weekly DF and DHF frequency tables as NumPy arrays."""
        freq_df, freq_dhf = self.build_weekly_frequency_tables()
        return (
            freq_df[["week", "count"]].to_numpy(dtype=int),
            freq_dhf[["week", "count"]].to_numpy(dtype=int),
        )

    def plot_weekly_frequency_data(self):
        """Display a quick plot of the computed weekly DF and DHF counts."""
        weekly_df_frequency_array, weekly_dhf_frequency_array = \
            self.build_weekly_frequency_arrays()
        plt.plot(weekly_df_frequency_array[2: -1, 0],
                 weekly_df_frequency_array[2: -1, 1],
                 ls='--',
                 marker='o',
                 mfc='blue',
                 ms=10,
                 alpha=0.4
                 )
        plt.plot(weekly_dhf_frequency_array[2: -1, 0],
                 weekly_dhf_frequency_array[2: -1, 1],
                 linestyle='-.',
                 marker='o',
                 mfc='red',
                 ms=10,
                 alpha=0.4
                 )
        plt.show()
