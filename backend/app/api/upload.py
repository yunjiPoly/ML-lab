"""Bounded, in-memory multipart reading for the recognize endpoint.

Starlette's form parser spools file parts larger than 1 MB to a temporary
file on disk.  Uploaded photos must never touch the disk, so this module feeds
the request body straight into the streaming ``python_multipart`` parser and
keeps the wanted file part in memory only.  The byte limit is enforced while
the body is still streaming: an oversize upload is rejected (413) before it
has been read completely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.requests import Request

from app.services.recognition.image_io import UploadValidationError

logger = logging.getLogger(__name__)

MULTIPART_OVERHEAD_BYTES = 64 * 1024
"""Allowance for boundaries, part headers and small extra fields on top of the file limit."""


@dataclass
class UploadedFile:
    """One file part of a multipart request, fully in memory."""

    field_name: str
    filename: str | None
    content_type: str | None
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


def _decode(value: bytes | bytearray | None) -> str | None:
    if not value:
        return None
    return bytes(value).decode("latin-1")


class _PartCollector:
    """python_multipart callbacks that keep only the wanted file field in memory."""

    def __init__(self, wanted_field: str, max_bytes: int) -> None:
        self._wanted = wanted_field
        self._max_bytes = max_bytes
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._buffer: bytearray | None = None
        self._filename: str | None = None
        self._content_type: str | None = None
        self.found: UploadedFile | None = None

    # -- callbacks ---------------------------------------------------------
    def on_part_begin(self) -> None:
        self._headers = {}
        self._buffer = None
        self._filename = None
        self._content_type = None

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        self._headers[bytes(self._header_field).lower()] = bytes(self._header_value)
        self._header_field = bytearray()
        self._header_value = bytearray()

    def on_headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = _decode(params.get(b"name"))
        if name != self._wanted or self.found is not None:
            return
        self._filename = _decode(params.get(b"filename"))
        self._content_type = _decode(self._headers.get(b"content-type"))
        self._buffer = bytearray()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._buffer is None:
            return
        self._buffer += data[start:end]
        if len(self._buffer) > self._max_bytes:
            raise UploadValidationError(413, f"upload exceeds the limit of {self._max_bytes} bytes")

    def on_part_end(self) -> None:
        if self._buffer is None:
            return
        self.found = UploadedFile(
            field_name=self._wanted, filename=self._filename, content_type=self._content_type, data=bytes(self._buffer)
        )
        self._buffer = None

    def on_end(self) -> None:
        return None

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self.on_part_begin,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_end": self.on_end,
        }


def _boundary(request: Request) -> bytes:
    media_type, params = parse_options_header(request.headers.get("content-type", ""))
    if media_type != b"multipart/form-data" or not params.get(b"boundary"):
        raise UploadValidationError(415, "expected a multipart/form-data request with an 'image' file field")
    return params[b"boundary"]


async def read_upload(request: Request, *, field: str, max_bytes: int) -> UploadedFile:
    """Read the file part ``field`` of a multipart request into memory.

    Raises :class:`UploadValidationError` (413) as soon as the body or the file
    part exceeds ``max_bytes``, (415) when the request is not multipart, (400)
    when the multipart body is malformed and (422) when the field is missing.
    """
    boundary = _boundary(request)
    body_limit = max_bytes + MULTIPART_OVERHEAD_BYTES
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > body_limit:
        raise UploadValidationError(413, f"request body of {declared} bytes exceeds the upload limit of {max_bytes} bytes")

    collector = _PartCollector(field, max_bytes)
    parser = MultipartParser(boundary, collector.callbacks())
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > body_limit:
                raise UploadValidationError(413, f"request body exceeds the upload limit of {max_bytes} bytes")
            parser.write(chunk)
        parser.finalize()
    except UploadValidationError:
        raise
    except Exception as exc:
        logger.info("Malformed multipart body: %s", exc)
        raise UploadValidationError(400, "malformed multipart body") from exc
    if collector.found is None:
        raise UploadValidationError(422, f"multipart file field '{field}' is required")
    return collector.found
