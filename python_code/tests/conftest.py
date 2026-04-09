import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tests")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def sample_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sqlite3.connect(data_dir / "dengue_data_2010.sqlite").close()

    np.savetxt(
        data_dir / "frecuency_per_week_DF.dat",
        np.array([[1, 2], [2, 1]]),
        fmt="%d",
        delimiter=",",
    )
    np.savetxt(
        data_dir / "frecuency_per_week_DHF.dat",
        np.array([[1, 1], [2, 1]]),
        fmt="%d",
        delimiter=",",
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
