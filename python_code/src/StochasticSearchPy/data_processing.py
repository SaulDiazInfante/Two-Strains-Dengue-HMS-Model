"""Data preparation utilities for the dengue model package."""

import numpy as np
import datetime
import sqlite3
import matplotlib.pyplot as plt
import csv
from sqlite3 import OperationalError
import os
from collections import Counter
from pathlib import Path
import pandas as pd

try:
    from .data_assets import resolve_data_dir
except ImportError:
    from data_assets import resolve_data_dir


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
        # self.data_DF = np.loadtxt('./raw_data/dengue_c_her2010.dat')
        # self.data_DHF = np.loadtxt('./raw_data/dengue_h_her2010.dat')
        # self.dates_excel_DF = self.data_DF[:, 0]
        # self.dates_excel_DHF = self.data_DHF[:, 1]
        # self.day_zero_classic_excel = 40386
        # self.day_zero_hemorrhagic_excel = 40386
        self.result_DF = []
        self.result_DHF = []
        self.date_frecuency_DF = []
        self.date_frecuency_DHF = []
        self.frecuency_per_week_DF = []
        self.frecuency_per_week_DHF = []
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = resolve_data_dir(data_dir, create=data_dir is not None)
        # The original script expected to run a setup script under ./raw_data, but
        # that file is not present in this repository. Guard the call to avoid
        # crashes while keeping the rest of the pipeline functional.
        setup_script = self.data_dir / "setup.sh"
        if setup_script.exists():
            os.system(f"sh {setup_script}")
        self.connection = sqlite3.connect(self._data_path('dengue_data_2010.sqlite'))
        self.cur = self.connection.cursor()

    def _data_path(self, filename: str) -> str:
        """Return the absolute path for a file inside the configured data directory."""
        return str(self.data_dir / filename)

    def close(self):
        """Close the underlying SQLite connection if it is open."""
        if getattr(self, "connection", None) is not None:
            self.connection.close()

    def __enter__(self):
        """Return the processor so it can be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close database resources when leaving a context manager block."""
        self.close()

    @staticmethod
    def excel_date2python_datetime(excel_date):
        """Convert an Excel day offset into a Python ``datetime``."""
        temp = datetime.datetime(1900, 1, 1)
        delta = datetime.timedelta(days=excel_date)
        return temp + delta

    def excel_data_dates2python_datetime(self, dates_excel):
        """Convert an iterable of Excel day offsets into Python datetimes."""
        dates_py = []
        for day in dates_excel:
            date_py = self.excel_date2python_datetime(day)
            dates_py.append(date_py)
        return dates_py

    @staticmethod
    def indices(lst, element_to_match):
        """Return all indices where ``element_to_match`` occurs in ``lst``."""
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element_to_match, offset + 1)
            except ValueError:
                return result
            result.append(offset)

    def create_nominal_data(self):
        """Populate the ``nominal_data`` SQLite table from the reference CSV.

        If the table already exists and contains rows, it is reused as-is.
        """
        # If the nominal_data table already contains rows, reuse it instead of
        # requiring the CSV file (which is not present in this repo).
        self.cur.execute("SELECT count(*) FROM sqlite_master "
                         "WHERE type='table' AND name='nominal_data';")
        table_exists = self.cur.fetchone()[0] == 1
        if table_exists:
            self.cur.execute("SELECT COUNT(*) FROM nominal_data;")
            if self.cur.fetchone()[0] > 0:
                return None

        # Fallback: build the table from the CSV if available.
        csv_path = Path(self._data_path('casos_dengue_2010_AGEB2010.csv'))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing raw_data file {csv_path}. Provide the CSV or an "
                "already-populated nominal_data table."
            )

        try:
            self.cur.execute("CREATE TABLE nominal_data (\
                        OBJECTID, Join_Count, TARGET_FID, ID_HILLO, ID_SON,\
                        ANYO,\
                        FOLIO, SEXO, EDAD, D_H_, U_NOTIFICA, J_N_, DOMICILIO, \
                        COLONIA, LOCALIDAD, MUNICIPIO, I_CUADRO, I_CUADRO2, \
                        S_E__INICI, J_S_, HOSP, HEMORRAGIA, PLAQUETAS,\
                        DEFUNCION, FOLIO_DENG, DIAS_EVOLU,\
                        IgM_ELISA, IgG_ELISA, NS1, AISLAMIENT, FD_FHD,\
                        DIAGNOSTIC,\
                        OBSERVACIO, ID_HILLO_1, EDAD2, CUANTIFICA, GRAVE, \
                        CVEGEO,\
                        CVEGEO_1, POB1, POB26_R, POB27_R, POB30_R, ECO25_R,\
                        SALUD2_R, EDU34_R, MIG11_R, VIV2_R, VIV4_R, VIV9_R, \
                        VIV17_R, VIV26_R, VIV27_R, VIV28_R, VIV31_R, VIV32_R, \
                        VIV33_R, VIV34_R, VIV35_R, VIV36_R, GRADO_MARG, \
                        GRADO_MA_1, hectarea, DENSI_POB, VIV2, INDEX_NORM, \
                        INDEX_NO_1, x, y);")
        except OperationalError:
            None
        with open(csv_path, 'r', newline='') as file_in:
            data_record_lines = csv.DictReader(file_in)
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
                for i in data_record_lines
                ]
        self.cur.executemany("INSERT INTO nominal_data(\
                    OBJECTID, Join_Count, TARGET_FID, ID_HILLO, ID_SON,\
                    ANYO, FOLIO, SEXO, EDAD, D_H_, U_NOTIFICA, J_N_, \
                    DOMICILIO, COLONIA, LOCALIDAD, MUNICIPIO, I_CUADRO, \
                    I_CUADRO2, S_E__INICI, J_S_, HOSP, HEMORRAGIA, \
                    PLAQUETAS, DEFUNCION, FOLIO_DENG, DIAS_EVOLU,\
                    IgM_ELISA, IgG_ELISA, NS1, AISLAMIENT, FD_FHD, DIAGNOSTIC,\
                    OBSERVACIO, ID_HILLO_1, EDAD2, CUANTIFICA, GRAVE, CVEGEO,\
                    CVEGEO_1, POB1, POB26_R, POB27_R, POB30_R, ECO25_R, \
                    SALUD2_R, EDU34_R, MIG11_R, VIV2_R, VIV4_R, VIV9_R, \
                    VIV17_R, VIV26_R, VIV27_R, VIV28_R, VIV31_R, VIV32_R, \
                    VIV33_R, VIV34_R, VIV35_R, VIV36_R, GRADO_MARG,\
                    GRADO_MA_1, hectarea, DENSI_POB, VIV2, INDEX_NORM, \
                    INDEX_NO_1, x, y)\
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,\
                            ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
        self.connection.commit()
        return None

#    @staticmethod
    def create_table_incidence_data(self):
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
            with open(self._data_path('casos_dengue_2010_AGEB2010.csv'), 'r', newline='') as file_in:
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

    def hemorrhagic_query_date_week(self):
        """Export DF and DHF incidence rows including date and epidemiological week."""
        self.create_nominal_data()
        self.create_table_incidence_data()
        query_buffer_DF = "SELECT case_date, case_week, \
                            fever FROM incidence_data WHERE fever = 'FD'"
        self.cur.execute(query_buffer_DF)
        result_DF = self.cur.fetchall()
        with open(self._data_path('incidence_data_DF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DF)
        #
        # frecuency per day counting
        #
        query_buffer_DHF = "SELECT case_date, case_week, \
                           fever FROM incidence_data WHERE fever = 'FHD'"
        self.cur.execute(query_buffer_DHF)
        result_DHF = self.cur.fetchall()
        with open(self._data_path('incidence_data_DHF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DHF)

        self.result_DHF = result_DHF
        self.result_DF = result_DF
        return result_DF, result_DHF
    
    def hemorrhagic_query(self):
        """Export DF and DHF incidence rows keyed only by case date."""
        self.create_nominal_data()
        self.create_table_incidence_data()
        query_buffer_DF = "SELECT case_date, \
                            fever FROM incidence_data WHERE fever = 'FD'"
        self.cur.execute(query_buffer_DF)
        result_DF = self.cur.fetchall()
        with open(self._data_path('incidence_data_DF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DF)
        #
        # frecuency per day counting
        #
        query_buffer_DHF = "SELECT case_date, \
                           fever FROM incidence_data WHERE fever = 'FHD'"
        self.cur.execute(query_buffer_DHF)
        result_DHF = self.cur.fetchall()
        with open(self._data_path('incidence_data_DHF.csv'), 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result_DHF)

        self.result_DHF = result_DHF
        self.result_DF = result_DF
        return result_DF, result_DHF
    
    def frecuency_per_day_and_week(self):
        """Aggregate raw case rows into weekly DF and DHF count tables.

        The method writes both intermediate per-day files and the weekly
        ``.dat`` tables consumed by :class:`StochasticSearch`.
        """
        result_DF, result_DHF = self.hemorrhagic_query()
        occurrences_dates_DF = [result_DF[i][0] for i in
                                range(1, len(result_DF))]
        occurrences_dates_DHF = [result_DHF[i][0] for i in
                                 range(1, len(result_DHF))]

        frecuency_per_date_DF = Counter(occurrences_dates_DF)
        frecuency_per_date_DHF = Counter(occurrences_dates_DHF)
        frecuency_per_date_DF_list = []
        frecuency_per_date_DHF_list = []
        # Change to list
        for key, value in frecuency_per_date_DF.items():
            temp = [key, value]
            frecuency_per_date_DF_list.append(temp)

        for key, value in frecuency_per_date_DHF.items():
            temp = [key, value]
            frecuency_per_date_DHF_list.append(temp)
        frecuency_per_date_week_DF_list = []
        frecuency_per_date_week_DHF_list = []
        # adding headers
        # frecuency_per_date_week_DF_list.append(['date', 'week', 'frecuency'])
        # frecuency_per_date_week_DHF_list.append(['date', 'week',
        # 'frecuency'])
        for element in frecuency_per_date_DF_list:
            dt = element[0]
            dt = datetime.datetime.strptime(dt, "%m/%d/%Y").date()
            week = dt.strftime('%U')
            frecuency_per_date_week_DF_list.append([element[0], week,
                                                    element[1]])
        for element in frecuency_per_date_DHF_list:
            dt = element[0]
            dt = datetime.datetime.strptime(dt, "%m/%d/%Y").date()
            week = dt.strftime('%U')
            frecuency_per_date_week_DHF_list.append([element[0], week,
                                                     element[1]])
        frecuency_per_date_week_DF_list.sort()
        frecuency_per_date_week_DHF_list.sort()
        f1 = self._data_path('frecuency_per_day_per_week_FD.dat')
        f2 = self._data_path('frecuency_per_day_per_week_FHD.dat')
        #
        with open(f1, 'w') as file_handler:
            file_handler.seek(0)
            file_handler.write(
                "{},{},{}\n".format('date', 'week', 'frecuency'))
            for item in frecuency_per_date_week_DF_list:
                file_handler.write(
                    "{},{},{}\n".format(item[0], item[1], item[2]))
        file_handler.close()
        #
        with open(f2, 'w') as file_handler:
            file_handler.seek(0)
            file_handler.write(
                "{},{},{}\n".format('date', 'week', 'frecuency'))
            for item in frecuency_per_date_week_DHF_list:
                file_handler.write(
                    "{},{},{}\n".format(item[0], item[1], item[2]))
        file_handler.close()

        frecuency_week_DF = [item[1] for item in frecuency_per_date_week_DF_list]
        frecuency_week_DHF = [item[1] for item in frecuency_per_date_week_DHF_list]

        # Aggregate counts directly per epidemiological week.
        week_counts_DF = Counter(frecuency_week_DF)
        week_counts_DHF = Counter(frecuency_week_DHF)

        frecuency_per_week_DF = np.array(
            sorted([[int(week), count] for week, count in week_counts_DF.items()],
                   key=lambda x: x[0])
        )
        frecuency_per_week_DHF = np.array(
            sorted([[int(week), count] for week, count in week_counts_DHF.items()],
                   key=lambda x: x[0])
        )
        self.frecuency_per_week_DF = frecuency_per_week_DF
        self.frecuency_per_week_DHF = frecuency_per_week_DHF
        np.savetxt(self._data_path('frecuency_per_week_DF.dat'),
                   frecuency_per_week_DF, fmt='%d', delimiter=',')
        np.savetxt(self._data_path('frecuency_per_week_DHF.dat'),
                   frecuency_per_week_DHF, fmt='%d', delimiter=',')

        return frecuency_per_week_DF, frecuency_per_week_DHF

    def plot_data_frecuency_per_week(self):
        """Display a quick plot of the computed weekly DF and DHF counts."""
        frecuency_per_week_DF, frecuency_per_week_DHF = \
            self.frecuency_per_day_and_week()
        plt.plot(frecuency_per_week_DF[2: -1, 0],
                 frecuency_per_week_DF[2: -1, 1],
                 ls='--',
                 marker='o',
                 mfc='blue',
                 ms=10,
                 alpha=0.4
                 )
        plt.plot(frecuency_per_week_DHF[2: -1, 0],
                 frecuency_per_week_DHF[2: -1, 1],
                 linestyle='-.',
                 marker='o',
                 mfc='red',
                 ms=10,
                 alpha=0.4
                 )
        plt.show()

    def incidence_frequency_tables(self):
        """Build weekly frequency tables for DF and DHF from incidence CSVs.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Two tables with ``week`` and ``count`` columns for DF and DHF.
        """
        date_cols = ['date', 'label']
        df_df = pd.read_csv(
            self._data_path('incidence_data_DF.csv'),
            header=None,
            names=date_cols,
            parse_dates=['date'],
            dayfirst=False,
        )
        df_dhf = pd.read_csv(
            self._data_path('incidence_data_DHF.csv'),
            header=None,
            names=date_cols,
            parse_dates=['date'],
            dayfirst=False,
        )

        def _week_counts(frame: pd.DataFrame) -> pd.DataFrame:
            week = frame['date'].dt.isocalendar().week
            counts = frame.groupby(week).size().reset_index(name='count')
            counts.rename(columns={'week': 'week'}, inplace=True)
            return counts

        freq_df = _week_counts(df_df)
        freq_dhf = _week_counts(df_dhf)


        freq_df.to_csv(self._data_path('frecuency_per_week_DF.csv'),
                       index=False)
        freq_dhf.to_csv(self._data_path('frecuency_per_week_DHF.csv'),
                        index=False)

        return freq_df, freq_dhf
