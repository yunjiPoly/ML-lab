# Legacy prototype scripts

These are the original exploration scripts that preceded the `backend/` application.
They are kept for reference only and are **superseded**:

| Script | Superseded by |
| --- | --- |
| `download_cards.py` | `python -m app.cli sync-cards` (`backend/app/services/providers/ygoprodeck.py`, `backend/app/services/sync.py`) |
| `process_photo.py` | `backend/app/services/card_detection/` (detector, warp, orientation, service) |
| `crop_regions.py` | `backend/app/core/layout.py` + `backend/app/services/card_detection/roi.py` |
| `test_paddle_ocr.py` | `backend/app/services/ocr/paddle_provider.py` |
| `generate_embeddings.py` | Seed for the future visual fallback (`backend/app/services/visual/`, currently `NoOpVisualRecognizer`). Uses CLIP; the planned implementation uses DINOv2 + NumPy/FAISS. |

They expect to be run from the repository root with the old `data/cards.json`
catalog and are not maintained.
