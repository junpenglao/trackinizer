"""Distribution metadata regressions for the Trax command."""

from pathlib import Path

import tomllib


def test_distribution_installs_trax_console_script() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["scripts"]["trax"] == "trackinizer.trax.cli:main"
