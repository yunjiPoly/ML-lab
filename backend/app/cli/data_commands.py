"""Data management CLI commands: ``init-db``, ``sync-cards``, ``search``, ``stats``.

Registered on the shared typer app by :func:`register` (see ``app.cli.__main__``).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from app.core.config import Settings, get_settings
from app.db.repository import SQLAlchemyCardRepository
from app.db.session import Database
from app.services.providers.base import CardDataProvider
from app.services.sync import DEV_CARD_NAMES, SyncReport, SyncService

logger = logging.getLogger(__name__)


class SyncMode(str, Enum):
    dev = "dev"
    full = "full"


def _open_database(settings: Settings) -> Database:
    return Database(settings.resolved_database_url, echo=settings.sql_echo)


def build_provider(settings: Settings, from_file: Path | None) -> CardDataProvider:
    """Offline JSON provider when ``from_file`` is given, otherwise the live YGOPRODeck API."""
    if from_file is not None:
        from app.services.providers.json_file import JsonFileProvider

        return JsonFileProvider(from_file)
    from app.services.providers.ygoprodeck import YGOProDeckProvider

    return YGOProDeckProvider(
        settings.ygoprodeck_base_url,
        request_delay_seconds=settings.ygoprodeck_request_delay_seconds,
        timeout_seconds=settings.ygoprodeck_timeout_seconds,
    )


def _echo_report(report: SyncReport, settings: Settings, database: Database) -> None:
    typer.echo("")
    typer.echo("Sync summary")
    typer.echo("-" * 40)
    typer.echo(report.summary())
    typer.echo("-" * 40)
    typer.echo(f"database:  {database.url}")
    typer.echo(f"image dir: {settings.resolved_card_image_dir}")


def register(app: typer.Typer) -> None:
    """Attach the data commands to ``app``."""

    @app.command("init-db")
    def init_db() -> None:
        """Create the database tables (idempotent) and print the database URL."""
        settings = get_settings()
        database = _open_database(settings)
        try:
            database.create_all()
        finally:
            database.dispose()
        typer.echo(f"Database ready: {database.url}")

    @app.command("sync-cards")
    def sync_cards(
        mode: Annotated[
            SyncMode, typer.Option("--mode", case_sensitive=False, help="dev: the development card list; full: whole catalog.")
        ] = SyncMode.dev,
        names: Annotated[
            list[str] | None, typer.Option("--name", help="Card name to sync (repeatable; overrides the dev list).")
        ] = None,
        from_file: Annotated[
            Path | None,
            typer.Option("--from-file", exists=True, dir_okay=False, readable=True, help="Offline JSON catalog instead of the API."),
        ] = None,
        images: Annotated[
            bool | None, typer.Option("--images/--no-images", help="Download artwork images (default: yes for dev, no for full).")
        ] = None,
        limit: Annotated[int | None, typer.Option("--limit", min=1, help="Full mode: stop after this many cards.")] = None,
    ) -> None:
        """Fetch card data from a provider and upsert it into the database."""
        settings = get_settings()
        database = _open_database(settings)
        provider = build_provider(settings, from_file)
        download_images = images if images is not None else mode is SyncMode.dev
        service = SyncService(database, provider, settings)
        typer.echo(f"Provider: {provider.name}  mode: {mode.value}  images: {'yes' if download_images else 'no'}")
        try:
            if mode is SyncMode.full and not names:
                report = service.sync_all(download_images=download_images, limit=limit)
            else:
                targets = list(names) if names else list(DEV_CARD_NAMES)
                typer.echo(f"Syncing {len(targets)} card name(s)...")
                report = service.sync_names(targets, download_images=download_images)
        finally:
            database.dispose()
        _echo_report(report, settings, database)
        if report.errors and report.cards_seen == 0:
            raise typer.Exit(code=1)

    @app.command("search")
    def search(
        query: Annotated[str, typer.Argument(help="Card name (or part of it).")],
        limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum number of results.")] = 10,
    ) -> None:
        """Search cards by name (prefix matches first, then substring matches)."""
        settings = get_settings()
        database = _open_database(settings)
        try:
            with database.session() as session:
                repository = SQLAlchemyCardRepository(session)
                cards = repository.search_cards(query, limit=limit)
                if not cards:
                    typer.echo(f"No cards matching {query!r}.")
                    return
                typer.echo(f"{len(cards)} result(s) for {query!r}:")
                for card in cards:
                    printings = repository.printings_for_card(card.id)
                    codes = ", ".join(p.set_code for p in printings[:6])
                    more = f" (+{len(printings) - 6} more)" if len(printings) > 6 else ""
                    typer.echo(f"  [{card.id}] {card.name}  ({card.type or '?'}; {len(printings)} printing(s): {codes}{more})")
        finally:
            database.dispose()

    @app.command("stats")
    def stats() -> None:
        """Print row counts for cards, printings and artworks."""
        settings = get_settings()
        database = _open_database(settings)
        try:
            database.create_all()
            with database.session() as session:
                repository = SQLAlchemyCardRepository(session)
                typer.echo(f"database:            {database.url}")
                typer.echo(f"cards:               {repository.count_cards()}")
                typer.echo(f"printings:           {repository.count_printings()}")
                typer.echo(f"artworks:            {repository.count_artworks()}")
                typer.echo(f"artworks with image: {repository.count_artworks_with_local_image()}")
                typer.echo(f"image dir:           {settings.resolved_card_image_dir}")
        finally:
            database.dispose()
