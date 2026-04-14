try:
    import _bootstrap  # noqa: F401 - ensures src/ is on sys.path for unittest runs
except ImportError:
    from . import _bootstrap  # noqa: F401 - package-relative fallback

from pathlib import Path

import yaml


def test_repo_root_workflows_exist_and_parse():
    repo_root = Path(__file__).resolve().parents[2]
    ci_workflow = repo_root / ".github" / "workflows" / "ci.yml"
    pages_workflow = repo_root / ".github" / "workflows" / "pages.yml"

    assert ci_workflow.exists()
    assert pages_workflow.exists()

    ci = yaml.safe_load(ci_workflow.read_text(encoding="utf-8"))
    pages = yaml.safe_load(pages_workflow.read_text(encoding="utf-8"))

    assert "jobs" in ci
    assert "jobs" in pages


def test_repo_root_workflows_reference_python_code_subdirectory():
    repo_root = Path(__file__).resolve().parents[2]
    ci_contents = (repo_root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    pages_contents = (repo_root / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )

    assert "working-directory: python_code" in ci_contents
    assert "python_code/site" in pages_contents
