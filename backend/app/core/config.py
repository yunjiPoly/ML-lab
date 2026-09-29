"""Application configuration.

All tunable values live here and are overridable through environment variables
or a ``.env`` file at the repository root (see ``.env.example``).  Nothing in the
application logic should hard-code paths; use :class:`Settings` instead.

Path resolution: the repository root is derived from this file's location
(``<root>/backend/app/core/config.py``) so the defaults work from any current
working directory on Windows and Linux.  ``PROJECT_ROOT`` can be overridden.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT: Path = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings.  Field names map 1:1 to environment variables
    (case-insensitive), e.g. ``DATABASE_URL``, ``SAVE_DEBUG_IMAGES``."""

    model_config = SettingsConfigDict(
        env_file=(DEFAULT_PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Yu-Gi-Oh! Card Recognizer"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = Field(default=False, description="Emit JSON log lines instead of text.")

    # ---------------------------------------------------------------- paths
    project_root: Path = DEFAULT_PROJECT_ROOT
    data_dir: Path | None = Field(default=None, description="Defaults to <project_root>/data")
    debug_dir: Path | None = Field(default=None, description="Defaults to <data_dir>/debug")
    card_image_dir: Path | None = Field(default=None, description="Defaults to <data_dir>/cards")

    # ------------------------------------------------------------- database
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy URL. Defaults to a SQLite file under data_dir. "
        "Use e.g. postgresql+psycopg://user:pass@host/db later.",
    )
    sql_echo: bool = False

    # ------------------------------------------------------------------ api
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    cors_origin_regex: str | None = (
        r"^https?://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$"
    )
    max_upload_bytes: int = 12 * 1024 * 1024
    allowed_upload_mime_types: str = "image/jpeg,image/png,image/webp"
    save_debug_images: bool = Field(
        default=False,
        description="When true the API also writes debug images under debug_dir. "
        "Uploaded images are never persisted otherwise.",
    )

    # ---------------------------------------------------------- card layout
    card_width: int = 421
    card_height: int = 614
    card_layout: str = Field(default="standard", description="Key of the ROI template to use.")
    ocr_scale: int = Field(
        default=2,
        ge=1,
        le=4,
        description="The card is also warped at card_width*ocr_scale for higher-resolution OCR crops.",
    )
    allow_full_image_fallback: bool = Field(
        default=True,
        description="If no card contour is found but the whole image already has card proportions, "
        "treat the whole image as the card (useful for scans / pre-cropped images).",
    )
    detection_min_area_ratio: float = Field(default=0.05, description="Min contour area / image area.")

    # ------------------------------------------------------------------ ocr
    ocr_provider: str = Field(default="paddle", description="paddle | fake (tests)")
    ocr_device: str = Field(default="cpu", description="cpu | gpu")
    ocr_rec_model: str = Field(
        default="PP-OCRv6_medium_rec",
        description="PaddleOCR text-recognition model name (PP-OCRv6_medium_rec, PP-OCRv6_small_rec, ...).",
    )
    ocr_min_confidence: float = Field(default=0.30, description="Below this an OCR field is treated as unusable.")

    # ------------------------------------------------------------- resolver
    resolver_name_candidate_threshold: float = Field(
        default=0.70, description="Min fuzzy score (0..1) for a card name to be considered a candidate."
    )
    resolver_name_strong_threshold: float = Field(
        default=0.90, description="Fuzzy score (0..1) at which a name match alone is considered strong."
    )
    resolver_ambiguity_margin: float = Field(
        default=0.05, description="Two name candidates closer than this margin are ambiguous."
    )
    resolver_max_candidates: int = 5

    # --------------------------------------------------------------- visual
    visual_recognizer: str = Field(default="noop", description="noop | (future) dinov2")
    visual_trigger_below_confidence: float = Field(
        default=0.60, description="Invoke the visual recognizer only when the OCR-based score is below this."
    )

    # -------------------------------------------------------- data provider
    card_data_provider: str = "ygoprodeck"
    ygoprodeck_base_url: str = "https://db.ygoprodeck.com/api/v7"
    ygoprodeck_request_delay_seconds: float = Field(
        default=0.15, description="Politeness delay between consecutive API requests."
    )
    ygoprodeck_timeout_seconds: float = 60.0

    # ------------------------------------------------------- derived paths
    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_data_dir(self) -> Path:
        return (self.data_dir or self.project_root / "data").resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_debug_dir(self) -> Path:
        return (self.debug_dir or self.resolved_data_dir / "debug").resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_card_image_dir(self) -> Path:
        return (self.card_image_dir or self.resolved_data_dir / "cards").resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = self.resolved_data_dir / "yugioh.sqlite3"
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_upload_mime_type_set(self) -> set[str]:
        return {m.strip().lower() for m in self.allowed_upload_mime_types.split(",") if m.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
