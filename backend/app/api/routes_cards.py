"""Card lookup endpoints: ``GET /api/cards/{id}`` and ``GET /api/search?q=``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import RepositoryDep
from app.schemas.cards import CardDetailOut, CardSearchOut, CardSummaryOut

router = APIRouter(tags=["cards"])

MIN_QUERY_LENGTH = 2


@router.get(
    "/cards/{card_id}",
    response_model=CardDetailOut,
    summary="Card details with artworks and printings",
    responses={404: {"description": "Unknown card id"}},
)
def get_card(card_id: int, repository: RepositoryDep) -> CardDetailOut:
    card = repository.get_card(card_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"card {card_id} not found")
    return CardDetailOut.model_validate(card)


@router.get(
    "/search",
    response_model=CardSearchOut,
    summary="Search cards by name",
    responses={400: {"description": "Query shorter than 2 characters"}},
)
def search(
    repository: RepositoryDep,
    q: str = Query(..., description="Card name or part of it (at least 2 characters)."),
    limit: int = Query(20, ge=1, le=100),
) -> CardSearchOut:
    query = q.strip()
    if len(query) < MIN_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"query must be at least {MIN_QUERY_LENGTH} characters")
    cards = repository.search_cards(query, limit=limit)
    return CardSearchOut(query=query, results=[CardSummaryOut.model_validate(card) for card in cards])
