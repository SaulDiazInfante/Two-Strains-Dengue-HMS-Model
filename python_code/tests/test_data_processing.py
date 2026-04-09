from pathlib import Path

from StochasticSearchPy import DataProcessing


def test_incidence_frequency_tables_use_configured_data_dir(sample_data_dir):
    with DataProcessing(data_dir=sample_data_dir) as processor:
        freq_df, freq_dhf = processor.incidence_frequency_tables()

    assert freq_df["count"].tolist() == [2, 1]
    assert freq_dhf["count"].tolist() == [1, 1]
    assert Path(sample_data_dir / "frecuency_per_week_DF.csv").exists()
    assert Path(sample_data_dir / "frecuency_per_week_DHF.csv").exists()


def test_excel_date_conversion_returns_datetime():
    converted = DataProcessing.excel_date2python_datetime(10)
    assert converted.year == 1900
    assert converted.month == 1
    assert converted.day == 11
