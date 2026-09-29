"""``recognize`` CLI command: run the full pipeline on one photo and print a report.

Exit codes: ``0`` when the card was ``MATCHED``, ``2`` for every other
recognition status (low confidence, ambiguous, not found, not detected, OCR
failed) and for usage errors such as a missing file (typer convention),
``1`` when the image exists but cannot be decoded.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from app.core.config import Settings, get_settings
from app.db.session import Database
from app.schemas.recognition import OcrFieldResult, RecognitionResult, RecognitionStatus
from app.services.card_detection.debug import DebugImageWriter
from app.services.recognition.factory import build_recognition_service
from app.services.recognition.image_io import ImageDecodeError, downscale_to_max_side, load_image_file

logger = logging.getLogger(__name__)

EXIT_MATCHED = 0
EXIT_ERROR = 1
EXIT_NOT_MATCHED = 2


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Plain fixed-width text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def _ocr_lines(label: str, field: OcrFieldResult | None) -> list[str]:
    if field is None:
        return [f"  {label:<10} (not read)"]
    lines = [f"  {label:<10} raw={field.raw!r}  confidence={field.confidence:.3f}  variant={field.variant or '-'}"]
    lines.append(f"  {'':<10} normalized={field.normalized!r}")
    if field.alternatives:
        shown = ", ".join(field.alternatives[:8])
        more = f" (+{len(field.alternatives) - 8} more)" if len(field.alternatives) > 8 else ""
        lines.append(f"  {'':<10} alternatives={shown}{more}")
    return lines


def format_report(result: RecognitionResult, image_path: Path) -> str:
    """Human-readable multi-section report of a :class:`RecognitionResult`."""
    out: list[str] = [f"Image: {image_path}", ""]

    detection = result.detection
    out.append("Detection")
    if detection is None or not detection.detected:
        out.append("  card detected: no")
    else:
        corners = " ".join(f"({x:.0f},{y:.0f})" for x, y in (detection.corners or []))
        out.append(f"  card detected: yes  method={detection.method}  rotation={detection.rotation_applied} deg  layout={detection.layout}")
        if corners:
            out.append(f"  corners (TL TR BR BL): {corners}")
    out.append("")

    out.append("OCR")
    out.extend(_ocr_lines("name", result.ocr.name))
    out.extend(_ocr_lines("set code", result.ocr.set_code))
    out.append("")

    if result.candidates:
        out.append("Candidates")
        rows = [
            [
                str(i + 1),
                f"[{c.card.id}] {c.card.name}",
                c.printing.set_code if c.printing else "-",
                c.printing.set_name if c.printing else "-",
                (c.printing.rarity if c.printing and c.printing.rarity else "-"),
                f"{c.score:.3f}",
                ", ".join(c.reasons),
            ]
            for i, c in enumerate(result.candidates)
        ]
        out.append(_table(["#", "Card", "Set code", "Set name", "Rarity", "Score", "Reasons"], rows))
        out.append("")

    out.append("Result")
    out.append(f"  status:     {result.status.value}")
    out.append(f"  card:       {f'[{result.card.id}] {result.card.name}' if result.card else '-'}")
    if result.printing:
        p = result.printing
        code = p.rarity_code or ""
        rarity = (f"{p.rarity}" + (f" ({code})" if code else "")) if p.rarity else "-"
        out.append(f"  printing:   {p.set_code} / {p.set_name} / {rarity}")
    else:
        out.append("  printing:   -")
    out.append(f"  confidence: {result.confidence:.3f} (application score, not a probability)")
    if result.notes:
        out.append("  notes:")
        out.extend(f"    - {note}" for note in result.notes)
    if result.timing_ms:
        out.append("  timing:     " + "  ".join(f"{k}={v:.0f}ms" for k, v in result.timing_ms.items()))
    if result.debug:
        out.append("")
        out.append("Debug images")
        out.extend(f"  {key:<28} {path}" for key, path in result.debug.items())
    return "\n".join(out)


def _run(settings: Settings, image_path: Path, writer: DebugImageWriter | None) -> RecognitionResult:
    database = Database(settings.resolved_database_url, echo=settings.sql_echo)
    try:
        database.create_all()
        service = build_recognition_service(settings, database)
        if len(service.name_index) == 0:
            typer.echo(
                f"warning: the database {database.url} has no cards; run 'sync-cards' first.", err=True
            )
        image = downscale_to_max_side(load_image_file(image_path), settings.max_image_side)
        return service.recognize(image, debug=writer)
    finally:
        database.dispose()


def register(app: typer.Typer) -> None:
    """Attach the ``recognize`` command to ``app``."""

    @app.command("recognize")
    def recognize(
        image_path: Annotated[
            Path,
            typer.Argument(exists=True, dir_okay=False, readable=True, help="Photo of one card (JPEG/PNG/WEBP)."),
        ],
        debug: Annotated[
            bool, typer.Option("--debug", help="Write intermediate images (contour, normalized card, ROIs, OCR variants).")
        ] = False,
        debug_dir: Annotated[
            Path | None, typer.Option("--debug-dir", help="Where to write debug images (default: <debug_dir>/<image stem>).")
        ] = None,
        as_json: Annotated[bool, typer.Option("--json", help="Print the RecognitionResult as JSON instead of a report.")] = False,
    ) -> None:
        """Recognize the card in IMAGE_PATH.  Exit code 0 when MATCHED, 2 otherwise, 1 on errors."""
        settings = get_settings()
        writer: DebugImageWriter | None = None
        if debug:
            writer = DebugImageWriter(debug_dir or settings.resolved_debug_dir / image_path.stem)
        try:
            result = _run(settings, image_path, writer)
        except (ImageDecodeError, FileNotFoundError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=EXIT_ERROR) from exc
        if as_json:
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(format_report(result, image_path))
        raise typer.Exit(code=EXIT_MATCHED if result.status is RecognitionStatus.MATCHED else EXIT_NOT_MATCHED)
