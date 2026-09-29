/**
 * Types for the card lookup / search endpoints
 * (see `backend/app/schemas/cards.py`) and the health endpoint.
 */
import type { CardSummary } from "./recognition";

export interface ArtworkOut {
  id: number;
  image_url: string | null;
  image_url_small: string | null;
  local_path: string | null;
}

export interface PrintingOut {
  id: number;
  set_code: string;
  set_name: string;
  rarity: string | null;
  rarity_code: string | null;
}

/** `GET /api/cards/{id}` */
export interface CardDetail extends CardSummary {
  description: string | null;
  race: string | null;
  attribute: string | null;
  level: number | null;
  atk: number | null;
  def: number | null;
  archetype: string | null;
  artworks: ArtworkOut[];
  printings: PrintingOut[];
}

/** `GET /api/search?q=` */
export interface CardSearchResponse {
  query: string;
  results: CardSummary[];
}

/** `GET /api/health` - the exact shape is owned by the backend; only `status` is assumed. */
export interface HealthResponse {
  status?: string;
  [key: string]: unknown;
}
