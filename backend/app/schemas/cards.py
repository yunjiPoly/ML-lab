"""API schemas for card lookup / search endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArtworkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_url: str | None = None
    image_url_small: str | None = None
    local_path: str | None = None


class PrintingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    set_code: str
    set_name: str
    rarity: str | None = None
    rarity_code: str | None = None


class CardSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    type: str | None = None
    frame_type: str | None = None


class CardDetailOut(CardSummaryOut):
    description: str | None = None
    race: str | None = None
    attribute: str | None = None
    level: int | None = None
    atk: int | None = None
    def_: int | None = Field(default=None, alias="def", serialization_alias="def")
    archetype: str | None = None
    artworks: list[ArtworkOut] = Field(default_factory=list)
    printings: list[PrintingOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class CardSearchOut(BaseModel):
    query: str
    results: list[CardSummaryOut]
