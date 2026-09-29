"""Engine and session factory.

SQLite for local development; the same code works with PostgreSQL by changing
``DATABASE_URL``.  No global mutable state: callers create a :class:`Database`
and pass it (or a session) to services.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_db_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create an engine with sensible defaults for the given URL."""
    connect_args: dict = {}
    if _is_sqlite(database_url):
        connect_args["check_same_thread"] = False
        # Make sure the parent directory of a file-based SQLite database exists.
        path_part = database_url.split("sqlite:///", 1)[-1]
        if path_part and path_part != ":memory:":
            Path(path_part).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, echo=echo, connect_args=connect_args, future=True)
    if _is_sqlite(database_url):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record) -> None:  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


class Database:
    """Owns an engine and a session factory."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self.url = database_url
        self.engine = create_db_engine(database_url, echo=echo)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def create_all(self) -> None:
        """Create tables that do not exist yet (idempotent)."""
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context-managed session: commits on success, rolls back on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
