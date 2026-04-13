from StochasticSearchPy import data_assets


def test_resolve_data_directory_prefers_environment_variable(tmp_path, monkeypatch):
    env_data_dir = tmp_path / "env-data"
    env_data_dir.mkdir()
    monkeypatch.setenv(data_assets.DATA_DIR_ENV_VAR, str(env_data_dir))

    assert data_assets.resolve_data_directory() == env_data_dir


def test_copy_reference_dataset_copies_reference_files(tmp_path, monkeypatch):
    source_dir = tmp_path / "reference"
    output_dir = tmp_path / "prepared"
    source_dir.mkdir()

    file_names = ("alpha.csv", "beta.dat")
    for file_name in file_names:
        (source_dir / file_name).write_text(file_name, encoding="utf-8")

    monkeypatch.setattr(data_assets, "REFERENCE_DATA_FILES", file_names)

    resolved_source, resolved_output, copied_files = data_assets.copy_reference_dataset(
        output_dir=output_dir,
        source_dir=source_dir,
    )

    assert resolved_source == source_dir
    assert resolved_output == output_dir
    assert [path.name for path in copied_files] == list(file_names)
    for file_name in file_names:
        assert (output_dir / file_name).read_text(encoding="utf-8") == file_name
