"""Shared pytest fixtures.

* ``fixtures_dir`` – path of ``tests/fixtures``.
* ``ygoprodeck_sample`` – parsed raw YGOPRODeck API response (dict with ``data``).
* ``memory_db`` – fresh in-memory SQLite :class:`Database` with tables created.
* ``test_settings`` – settings pointing at a temporary data directory.
* ``local_test_photo`` – path of the private real-card photo, or skips the test
  when it is not present (CI never depends on it).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import Settings, get_settings
from app.db.session import Database

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The known real-world development photo (Armades, Keeper of Boundaries / JOTL-EN045).
LOCAL_TEST_PHOTO_CANDIDATES = (
    PROJECT_ROOT / "data" / "test" / "card1.jpg",
    PROJECT_ROOT / "data" / "test" / "172026.jpg",
)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def ygoprodeck_sample(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "ygoprodeck_sample.json").read_text(encoding="utf-8"))


@pytest.fixture()
def memory_db() -> Iterator[Database]:
    db = Database("sqlite:///:memory:")
    db.create_all()
    try:
        yield db
    finally:
        db.dispose()


@pytest.fixture()
def test_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings isolated from the developer's .env and real data directory."""
    for key in list(os.environ):
        if key.upper() in {
            "DATABASE_URL", "DATA_DIR", "DEBUG_DIR", "SAVE_DEBUG_IMAGES", "OCR_PROVIDER", "CARD_LAYOUT",
        }:
            monkeypatch.delenv(key, raising=False)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        debug_dir=tmp_path / "debug",
        database_url=f"sqlite:///{(tmp_path / 'test.sqlite3').as_posix()}",
        ocr_provider="fake",
        save_debug_images=False,
    )
    get_settings.cache_clear()
    return settings


@pytest.fixture()
def local_test_photo() -> Path:
    for candidate in LOCAL_TEST_PHOTO_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.skip("Local test photo not present (data/test/card1.jpg or data/test/172026.jpg).")
