"""Entry point for ``python -m app.cli``.

Each command module exposes ``register(app: typer.Typer)`` so commands can be
developed independently.
"""

from __future__ import annotations

import typer

from app.core.config import get_settings
from app.core.logging import configure_logging

app = typer.Typer(
    help="Yu-Gi-Oh! card recognition toolkit.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG level logging."),
) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose or settings.debug else settings.log_level, settings.log_json)


def _register_all() -> None:
    from app.cli import data_commands, recognize_commands

    data_commands.register(app)
    recognize_commands.register(app)


_register_all()

if __name__ == "__main__":
    app()
