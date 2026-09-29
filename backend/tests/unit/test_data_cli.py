"""Typer CLI: init-db, sync-cards --from-file, stats, search against a temporary database."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from app.core.config import get_settings

runner = CliRunner()


@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    """The CLI app bound to a throw-away SQLite database and image directory.

    Environment variables are set *before* the CLI module is imported and the
    settings cache is cleared, so ``get_settings()`` picks them up.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.sqlite3').as_posix()}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CARD_IMAGE_DIR", str(tmp_path / "data" / "cards"))
    monkeypatch.setenv("OCR_PROVIDER", "fake")
    get_settings.cache_clear()
    try:
        from app.cli.__main__ import app as cli_app
    except ImportError:
        # Stage-2 command modules (recognize_commands) may not exist yet; the data
        # commands are self-contained, so register them on a bare app instead.
        from app.cli import data_commands

        cli_app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
        data_commands.register(cli_app)
    yield cli_app
    get_settings.cache_clear()


def test_init_db_creates_database(cli: typer.Typer, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["init-db"])
    assert result.exit_code == 0, result.output
    assert "Database ready: sqlite:///" in result.output
    assert (tmp_path / "cli.sqlite3").exists()


def test_sync_from_file_then_stats_and_search(cli: typer.Typer, fixtures_dir: Path, tmp_path: Path) -> None:
    fixture = fixtures_dir / "ygoprodeck_sample.json"
    result = runner.invoke(cli, ["sync-cards", "--from-file", str(fixture), "--no-images"])
    assert result.exit_code == 0, result.output
    assert "Provider: json_file" in result.output
    assert "created:         7" in result.output
    assert "images downloaded: 0" in result.output
    # 4 of the 11 dev names are not in the fixture -> reported, not fatal
    assert "card not found: 'Red-Eyes Black Dragon'" in result.output
    assert not (tmp_path / "data" / "cards").exists()

    stats = runner.invoke(cli, ["stats"])
    assert stats.exit_code == 0, stats.output
    assert "cards:               7" in stats.output
    assert "printings:           " in stats.output
    assert "artworks with image: 0" in stats.output

    search = runner.invoke(cli, ["search", "armades"])
    assert search.exit_code == 0, search.output
    assert "[88033975] Armades, Keeper of Boundaries" in search.output
    assert "JOTL-EN045" in search.output

    missing = runner.invoke(cli, ["search", "definitely-not-a-card"])
    assert missing.exit_code == 0
    assert "No cards matching" in missing.output


def test_sync_specific_names_from_file(cli: typer.Typer, fixtures_dir: Path) -> None:
    fixture = fixtures_dir / "ygoprodeck_sample.json"
    result = runner.invoke(cli, ["sync-cards", "--from-file", str(fixture), "--no-images", "--name", "Kuriboh", "--name", "Pot of Greed"])
    assert result.exit_code == 0, result.output
    assert "Syncing 2 card name(s)" in result.output
    assert "created:         2" in result.output
    assert "errors:            0" in result.output

    again = runner.invoke(cli, ["sync-cards", "--from-file", str(fixture), "--no-images", "--name", "Kuriboh"])
    assert again.exit_code == 0, again.output
    assert "unchanged:       1" in again.output


def test_sync_full_mode_from_file_with_limit(cli: typer.Typer, fixtures_dir: Path) -> None:
    fixture = fixtures_dir / "ygoprodeck_sample.json"
    result = runner.invoke(cli, ["sync-cards", "--mode", "full", "--from-file", str(fixture), "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert "images: no" in result.output  # full mode defaults to no images
    assert "cards seen:        3" in result.output


def test_sync_from_missing_file_fails(cli: typer.Typer, tmp_path: Path) -> None:
    result = runner.invoke(cli, ["sync-cards", "--from-file", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
