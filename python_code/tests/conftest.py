import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tests")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(name="sample_data_dir")
def build_sample_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with sqlite3.connect(data_dir / "dengue_data_2010.sqlite") as conn:
        conn.execute(
            """
            CREATE TABLE nominal_data (
                I_CUADRO TEXT,
                S_E__INICI TEXT,
                FD_FHD TEXT,
                x REAL,
                y REAL
            );
            """
        )
        conn.executemany(
            "INSERT INTO nominal_data (I_CUADRO, S_E__INICI, FD_FHD, x, y) VALUES (?, ?, ?, ?, ?);",
            [
                ("01/04/2010", "1", "FD", 0.0, 0.0),
                ("01/05/2010", "1", "FD", 0.0, 0.0),
                ("01/11/2010", "2", "FD", 0.0, 0.0),
                ("01/06/2010", "1", "FHD", 0.0, 0.0),
                ("01/12/2010", "2", "FHD", 0.0, 0.0),
            ],
        )

    (data_dir / "incidence_data_DF.csv").write_text(
        "01/04/2010,FD\n01/05/2010,FD\n01/11/2010,FD\n",
        encoding="utf-8",
    )
    (data_dir / "incidence_data_DHF.csv").write_text(
        "01/06/2010,FHD\n01/12/2010,FHD\n",
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture(name="real_data")
def copy_real_data_directory(tmp_path):
    """Return an isolated copy of the committed SQLite dataset for tests."""
    source_sqlite = (
        PROJECT_ROOT / "data" / "reference" / "raw_data" / "dengue_data_2010.sqlite"
    )
    if not source_sqlite.exists():
        pytest.skip(f"Missing committed reference SQLite dataset: {source_sqlite}")

    data_dir = tmp_path / "real-data"
    data_dir.mkdir()
    shutil.copy2(source_sqlite, data_dir / source_sqlite.name)
    return data_dir
