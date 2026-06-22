"""Utilities for launching the Streamlit interactive plotting frontend."""

import importlib.util
import subprocess
import sys
from pathlib import Path


def launch_interactive_plot_app(
    csv_path=None,
    index_column: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8501,
) -> int:
    """Launch the Streamlit time-series plotting app."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            "streamlit is not installed. Install UI extras with: "
            "python -m pip install -e '.[ui]'"
        )
        return 2

    app_path = Path(__file__).resolve().with_name("interactive_plot_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        str(host),
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
        "--",
    ]
    if csv_path is not None:
        command.extend(["--csv", str(Path(csv_path).expanduser())])
    if index_column:
        command.extend(["--index-column", str(index_column)])

    print(f"interactive_frontend_url=http://{host}:{port}")
    return subprocess.run(command, check=False).returncode
