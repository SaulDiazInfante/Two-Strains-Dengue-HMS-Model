from pathlib import Path

import yaml


def test_mkdocs_config_exists_and_points_to_docs():
    config_path = Path(__file__).resolve().parents[1] / "mkdocs.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["docs_dir"] == "docs"
    assert config["site_dir"] == "site"
    assert config["nav"]


def test_mkdocs_nav_targets_existing_docs_files():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "mkdocs.yml").read_text(encoding="utf-8"))

    for item in config["nav"]:
        _, relative_path = next(iter(item.items()))
        assert (project_root / "docs" / relative_path).exists()
