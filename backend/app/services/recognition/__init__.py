"""Stage-2 integration: the end-to-end recognition pipeline and its factory.

Public API::

    from app.services.recognition import RecognitionService, build_recognition_service
    from app.services.recognition.image_io import decode_image_bytes, load_image_file
"""

from app.services.recognition.factory import SessionScopedRepository, build_recognition_service
from app.services.recognition.image_io import (
    ImageDecodeError,
    UploadValidationError,
    decode_image_bytes,
    downscale_to_max_side,
    load_image_file,
    validate_upload,
)
from app.services.recognition.pipeline import RecognitionService

__all__ = [
    "ImageDecodeError",
    "RecognitionService",
    "SessionScopedRepository",
    "UploadValidationError",
    "build_recognition_service",
    "decode_image_bytes",
    "downscale_to_max_side",
    "load_image_file",
    "validate_upload",
]
