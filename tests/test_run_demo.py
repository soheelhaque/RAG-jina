"""Check that both public entry points work offline from another directory."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("entry_point", ["main.py", "scripts/run_demo.py"])
def test_entry_point_reports_whole_documents(entry_point: str, tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(project_root / entry_point)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Loaded 8 whole documents:" in result.stdout
    assert result.stdout.count("characters") == 8
    assert "01_equity_markets.txt:" in result.stdout
    assert "08_policy_and_regulation.txt:" in result.stdout
    assert "Documents remain whole" in result.stdout
