"""Helpers for locating and preparing model datasets.

The package itself ships code only. Reference data stays in the repository
under ``data/reference/raw_data`` and can be copied into a runtime directory
with :func:`copy_reference_dataset`.
"""

import os
import shutil
from pathlib import Path


DATA_DIR_ENV_VAR = "TWO_STRAINS_DENGUE_DATA_DIR"
REFERENCE_DATA_RELATIVE_PATH = Path("data/reference/raw_data")
CORE_MODEL_FILES = (
    "dengue_data_2010.sqlite",
)
REFERENCE_DATA_FILES = (
    "NOMINAL DENGUE 2015.xlsx",
    "NOMINALES DENGUE.zip",
    "cases_per_date.csv",
    "cases_per_date_2010.csv",
    "cases_per_date_2015.csv",
    "cases_per_date_FD.csv",
    "cases_per_date_FHD.csv",
    "casos_dengue_2010_AGEB2010.csv",
    "casos_dengue_2010_AGEB2010.dbf",
    "dengue_data_2010.sqlite",
    "frequency_per_week_DF.csv",
    "frequency_per_week_DHF.csv",
    "incidence_data.csv",
    "incidence_data_2010_fever_classification.csv",
    "incidence_data_2015.csv",
    "incidence_data_DF.csv",
    "incidence_data_DHF.csv",
    "incidence_data_FD.csv",
    "incidence_data_FHD.csv",
    "nominal_data_dengue_2010.csv",
    "nominal_data_dengue_2015.csv",
)


def get_repository_root() -> Path:
    """Return the repository root for a source checkout."""
    return Path(__file__).resolve().parents[2]


def get_reference_data_directory() -> Path:
    """Return the committed reference dataset directory."""
    return get_repository_root() / REFERENCE_DATA_RELATIVE_PATH


def get_default_data_directory() -> Path | None:
    """Resolve the default data directory for local runs.

    Resolution order:
    1. ``TWO_STRAINS_DENGUE_DATA_DIR``
    2. the committed reference dataset in a source checkout
    """
    env_data_dir = os.getenv(DATA_DIR_ENV_VAR)
    if env_data_dir:
        return Path(env_data_dir).expanduser()

    reference_dir = get_reference_data_directory()
    if reference_dir.exists():
        return reference_dir
    return None


def resolve_reference_source_directory(source_dir=None) -> Path:
    """Resolve the source directory used by ``prepare-data``.

    Parameters
    ----------
    source_dir:
        Explicit source directory. When omitted, the function expects a source
        checkout containing ``data/reference/raw_data``.
    """
    if source_dir is not None:
        return Path(source_dir).expanduser()

    reference_dir = get_reference_data_directory()
    if reference_dir.exists():
        return reference_dir

    raise FileNotFoundError(
        "No committed reference dataset was found. Pass `source_dir` pointing at "
        "`data/reference/raw_data` from a source checkout."
    )


def resolve_data_directory(data_dir=None, create=False) -> Path:
    """Resolve a runtime data directory for model execution.

    Parameters
    ----------
    data_dir:
        Explicit directory to use.
    create:
        When ``True``, create ``data_dir`` if it does not already exist.
    """
    if data_dir is not None:
        resolved = Path(data_dir).expanduser()
        if create and not resolved.exists():
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    resolved = get_default_data_directory()
    if resolved is not None:
        return resolved

    raise FileNotFoundError(
        "No data directory was found. Pass `data_dir`, set "
        f"`{DATA_DIR_ENV_VAR}`, or run "
        "`two-strains-dengue prepare-data --output-dir <path>` from a source checkout."
    )


def validate_required_files_exist(data_dir, file_names, label="data directory"):
    """Validate that a directory contains the expected data files."""
    data_path = Path(data_dir)
    missing_files = [name for name in file_names if not (data_path / name).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing files in {label} {data_path}: {missing}")
    return data_path


def copy_reference_dataset(output_dir, source_dir=None) -> tuple[Path, Path, list[Path]]:
    """Copy the committed reference dataset into a writable runtime directory.

    Parameters
    ----------
    output_dir:
        Destination directory that will receive the copied dataset.
    source_dir:
        Optional source directory. Defaults to the committed reference dataset
        in a source checkout.

    Returns
    -------
    tuple[Path, Path, list[Path]]
        The resolved source directory, resolved output directory, and copied
        file paths.
    """
    source_path = resolve_reference_source_directory(source_dir)
    validate_required_files_exist(
        source_path,
        REFERENCE_DATA_FILES,
        label="reference data directory",
    )

    target_path = Path(output_dir).expanduser()
    target_path.mkdir(parents=True, exist_ok=True)

    copied_paths = []
    for name in REFERENCE_DATA_FILES:
        copied = target_path / name
        shutil.copy2(source_path / name, copied)
        copied_paths.append(copied)

    return source_path, target_path, copied_paths
