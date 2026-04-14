try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from StochasticSearchPy import DataProcessing


def _count_expected_cases(data_dir, fever):
    sqlite_path = Path(data_dir) / "dengue_data_2010.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        cursor = connection.execute(
            "SELECT COUNT(*) FROM nominal_data WHERE FD_FHD = ?",
            (fever,),
        )
        return int(cursor.fetchone()[0])


def _build_expected_daily_counts(data_dir, fever):
    sqlite_path = Path(data_dir) / "dengue_data_2010.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        frame = pd.read_sql_query(
            "SELECT I_CUADRO AS date FROM nominal_data WHERE FD_FHD = ?",
            connection,
            params=(fever,),
        )

    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    expected = (
        frame.groupby(frame["date"].dt.normalize())
        .size()
        .reset_index(name="count")
        .sort_values("date")
        .reset_index(drop=True)
        .astype({"count": int})
    )
    expected.set_index("date", inplace=True)
    return expected


def _build_expected_weekly_counts(data_dir, fever):
    daily_counts = _build_expected_daily_counts(data_dir, fever)
    expected = (
        daily_counts.assign(
            week=daily_counts.index.to_series().dt.isocalendar().week.astype(int)
        )
        .groupby("week", as_index=False)["count"]
        .sum()
        .sort_values("week")
        .reset_index(drop=True)
        .astype({"week": int, "count": int})
    )
    return expected


def test_build_daily_frequency_tables_use_real_sqlite(real_data):
    with DataProcessing(data_dir=real_data) as processor:
        freq_df, freq_dhf = processor.build_daily_frequency_tables()

    expected_df = _build_expected_daily_counts(real_data, "FD")
    expected_dhf = _build_expected_daily_counts(real_data, "FHD")
    assert_frame_equal(freq_df, expected_df)
    assert_frame_equal(freq_dhf, expected_dhf)
    assert Path(real_data / "frequency_per_date_DF.csv").exists()
    assert Path(real_data / "frequency_per_date_DHF.csv").exists()


def test_build_weekly_frequency_tables_use_real_sqlite(real_data):
    with DataProcessing(data_dir=real_data) as processor:
        freq_df, freq_dhf = processor.build_weekly_frequency_tables()

    expected_df = _build_expected_weekly_counts(real_data, "FD")
    expected_dhf = _build_expected_weekly_counts(real_data, "FHD")
    assert_frame_equal(freq_df.astype({"week": int, "count": int}), expected_df)
    assert_frame_equal(freq_dhf.astype({"week": int, "count": int}), expected_dhf)
    assert Path(real_data / "frequency_per_week_DF.csv").exists()
    assert Path(real_data / "frequency_per_week_DHF.csv").exists()


def test_build_weekly_frequency_arrays_match_weekly_frequency_tables(real_data):
    with DataProcessing(data_dir=real_data) as processor:
        weekly_df, weekly_dhf = processor.build_weekly_frequency_tables()
        weekly_df_array, weekly_dhf_array = processor.build_weekly_frequency_arrays()

    assert weekly_df_array.tolist() == weekly_df[["week", "count"]].to_numpy(dtype=int).tolist()
    assert weekly_dhf_array.tolist() == weekly_dhf[["week", "count"]].to_numpy(dtype=int).tolist()


def test_excel_date_conversion_returns_datetime():
    converted = DataProcessing.convert_excel_date_to_datetime(10)
    assert converted.year == 1900
    assert converted.month == 1
    assert converted.day == 11


def test_real_data_fixture_exposes_committed_sqlite(real_data):
    sqlite_path = Path(real_data / "dengue_data_2010.sqlite")
    assert sqlite_path.exists()

    with DataProcessing(data_dir=real_data) as processor:
        assert processor.data_dir == real_data


def test_export_incidence_rows_with_weeks_creates_expected_output(real_data):
    with DataProcessing(data_dir=real_data) as processor:
        result_df, result_dhf = processor.export_incidence_rows_with_weeks()

    assert isinstance(result_df, list)
    assert isinstance(result_dhf, list)
    assert len(result_df) == _count_expected_cases(real_data, "FD")
    assert len(result_dhf) == _count_expected_cases(real_data, "FHD")
    assert isinstance(result_df[0], tuple)
    assert isinstance(result_dhf[0], tuple)
    assert len(result_df[0]) == 3
    assert len(result_dhf[0]) == 3
    assert Path(real_data / "incidence_data_DF.csv").exists()
    assert Path(real_data / "incidence_data_DHF.csv").exists()
