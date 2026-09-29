# Yu-Gi-Oh! Card Recognizer

Photograph a physical Yu-Gi-Oh! card with a phone (or upload a photo) and get back
the **card name**, the **exact printing** (set code such as `JOTL-EN045`), the
**set name**, the **rarity** and an honest **status / match score**.

```
Armades, Keeper of Boundaries
JOTL-EN045
Judgment of the Light
Secret Rare
MATCHED  (match score 0.9x)
```

The system is deliberately *not* an image classifier.  It is:

```
computer vision for geometry  +  OCR for text  +  database constraints
+  fuzzy matching  +  (optional) visual ML fallback
```

---

## 1. What it does

1. Detects the physical card in a real photo (busy playmat, sleeve, slight angle).
2. Perspective-corrects and normalizes it to a fixed upright size (421 x 614).
3. Crops only the regions that matter: the **card name** strip and the **set code**.
4. Runs local **PaddleOCR (PP-OCRv6 text recognition)** on those crops, trying a few
   preprocessing variants and keeping the most plausible reading.
5. Normalizes the text and resolves it against a local **SQLite** catalog
   (cards / artworks / printings synced from YGOPRODeck).
6. Returns a `RecognitionResult` with one of the statuses
   `MATCHED`, `LOW_CONFIDENCE`, `AMBIGUOUS`, `NOT_FOUND`, `CARD_NOT_DETECTED`, `OCR_FAILED`,
   plus candidates when ambiguity genuinely exists.

## 2. Architecture

```
Camera / upload (React + Vite frontend)
        |  POST /api/recognize (multipart)
        v
FastAPI backend (backend/app)
        |
        v
card_detection/   OpenCV: gray -> blur -> edges -> contours -> best 4-point quad
                  -> order corners -> perspective warp -> upright orientation
        |
        v
core/layout.py    ROI templates (name strip, set-code box) in canonical coordinates
        |
        v
ocr/              OCRProvider interface -> PaddleOCRProvider (recognition-only model,
                  several preprocessing variants per crop, best reading selected)
        |
        v
core/text.py      normalization (case / punctuation / dash / charset) and
                  *candidate* corrections for O/0, I/1, L/1, S/5, B/8 ...
        |
        v
resolver/         rapidfuzz name index + set-code lookup + decision rules
                  -> status, card, printing, candidates, match score
        |
        v
visual/           VisualRecognizer interface (NoOp today; DINOv2 + FAISS later),
                  only consulted when the OCR result is weak
```

Backend layout:

```
backend/
  app/
    api/                 FastAPI routers (no business logic)
    core/                config (env driven), logging, text normalization, ROI layouts
    db/                  engine/session + repository (SQLAlchemy 2.0)
    models/              cards / artworks / printings ORM
    schemas/             Pydantic contracts (RecognitionResult, card DTOs)
    services/
      card_detection/    detector, warp, orientation, roi, debug images
      ocr/               preprocessing, variants, selection, paddle/fake providers
      resolver/          name index, matching, resolver
      recognition/       pipeline orchestration + image I/O
      providers/         YGOPRODeck + offline JSON providers (pluggable)
      visual/            optional visual fallback interface
      sync.py            catalog synchronization
    cli/                 python -m app.cli ...
  tests/                 unit + integration (pytest)
frontend/                React + TypeScript + Vite scanner UI
data/                    local catalog DB, cached images, debug output, test photos
legacy/                  original prototype scripts (superseded, kept for reference)
```

### Why OCR instead of image classification

* A printing is identified by *text* that is literally printed on the card.
  Distinguishing `SDK-001` from `LOB-001` visually is nearly impossible; reading the
  code is trivial for OCR at the right crop resolution.
* The catalog changes constantly (new sets every few weeks).  A database sync is
  cheap; retraining a classifier is not.
* Reference images (YGOPRODeck) are artwork images, not photos of every physical
  printing, so an image classifier could never learn the exact-printing distinction.
* Visual embeddings remain useful as a *fallback* (damaged / foreign-language cards
  where OCR fails); the interface for that exists and is disabled by default.

### Why the set code matters

One card has many printings (Blue-Eyes White Dragon has 70+).  The set code
(`JOTL-EN045`) pins the exact product and, together with the rarity, the exact
version.  The name alone can only narrow candidates.  The tiny passcode at the
bottom of the card is deliberately **not** used: it is too small to OCR reliably
through sleeves and phone cameras.

### Data model

```
cards      (id, name, normalized_name*, type, frame_type, description, ...)
artworks   (id, card_id*, image_url, local_path)          one card -> many artworks
printings  (id, card_id*, set_code*, set_name, rarity, rarity_code)   one card -> many printings
* = indexed
```

Set codes are normalized (upper-case, `A-Z0-9-`) before persistence and lookup.

### Confidence

`confidence` in the API is an **application-level match score in [0, 1]**
combining OCR scores and database agreement.  It is *not* a calibrated
probability.

## 3. Development setup (Windows)

Prerequisites: Python 3.13 with the existing `.venv`, Node.js 18+ (24 tested).

```powershell
cd D:\LABS\yugioh-lab
.venv\Scripts\activate
pip install -r backend\requirements.txt
pip install -e backend --no-deps
copy .env.example .env      # optional; defaults work without it
```

`torch` / `transformers` are not needed for the MVP and are not reinstalled.
PaddleOCR downloads `PP-OCRv6_medium_rec` to `%USERPROFILE%\.paddlex\official_models`
on first use (a few MB).

## 4. Sync card data

Development (11 well-known cards + the Armades test card, with artwork images):

```powershell
python -m app.cli sync-cards --mode dev
```

Full catalog (about 13k cards, one large API request, no images by default):

```powershell
python -m app.cli sync-cards --mode full
```

Offline / from an existing JSON dump (raw YGOPRODeck response or the legacy `data/cards.json`):

```powershell
python -m app.cli sync-cards --from-file data\cards.json --no-images
```

Other data commands: `python -m app.cli init-db`, `python -m app.cli stats`,
`python -m app.cli search "blue-eyes"`.

The database is `data\yugioh.sqlite3` unless `DATABASE_URL` is set.  Re-running a
sync only rewrites cards whose content changed and never re-downloads existing images.

## 5. Start the backend

```powershell
cd D:\LABS\yugioh-lab
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* `GET  http://localhost:8000/api/health`
* `POST http://localhost:8000/api/recognize` (multipart field `image`)
* `GET  http://localhost:8000/api/cards/{id}`
* `GET  http://localhost:8000/api/search?q=armades`
* Interactive docs: `http://localhost:8000/docs`

## 6. Start the frontend

```powershell
cd D:\LABS\yugioh-lab\frontend
npm install
npm run dev
```

Open `http://localhost:5173`.  The dev server proxies `/api` to `http://localhost:8000`
so no CORS configuration is needed locally.

**Phone on the same Wi-Fi:** open `http://<your-pc-lan-ip>:5173`.  Browsers only
allow the live camera on HTTPS (or localhost), so either use the *Upload image*
mode (it opens the phone camera through the file picker) or start Vite with a
self-signed certificate:

```powershell
$env:VITE_HTTPS="true"; npm run dev
```

and open `https://<your-pc-lan-ip>:5173` (accept the certificate warning once).

## 7. Recognize a local image from the CLI

```powershell
python -m app.cli recognize data\test\172026.jpg --debug
```

Prints detection info, raw / normalized OCR values with confidences, candidate
matches and the final result.  With `--debug` it writes the detected contour,
normalized card, ROI overlays, name / set-code crops and every preprocessed OCR
variant under `data\debug\<image-name>\`.  `--json` prints the raw
`RecognitionResult`.  Exit code is `0` for `MATCHED`, `2` for any other status,
`1` if the image cannot be decoded.

Example output for the development photo:

```
OCR
  name       raw='ARMADES, KEEPER OF BOUNDARIES'  confidence=0.996  variant=tight_invert
  set code   raw='JOTL-EN045'  confidence=0.999  variant=up3_adaptive
Result
  status:     MATCHED
  card:       [88033975] Armades, Keeper of Boundaries
  printing:   JOTL-EN045 / Judgment of the Light / Secret Rare (ScR)
  confidence: 1.000 (application score, not a probability)
```

The first run in a process takes ~10 s (Paddle import + model load); afterwards a
photo takes roughly one second on CPU.

## 8. Tests

```powershell
python -m pytest backend\tests -q
```

Optional local-only tests (need the PaddleOCR model and the private test photo
`data\test\172026.jpg` or `data\test\card1.jpg`; skipped automatically otherwise):

```powershell
$env:RUN_OCR_TESTS="1"; python -m pytest backend\tests -m "integration or ocr" -q
```

## 9. Configuration

All settings are environment variables (see `.env.example`): `DATABASE_URL`,
`DEBUG`, `SAVE_DEBUG_IMAGES`, `CORS_ORIGINS`, `CORS_ORIGIN_REGEX`, `OCR_DEVICE`,
`OCR_REC_MODEL`, `OCR_WARMUP_ON_STARTUP`, `OCR_SCALE`, `CARD_LAYOUT`,
`MAX_UPLOAD_BYTES`, `MAX_IMAGE_SIDE`, resolver thresholds, ...
No cloud account is required.

By default the API loads the OCR model lazily on the first `/api/recognize`
request (about 10 s).  Set `OCR_WARMUP_ON_STARTUP=true` to load it in the
background when the server starts.

## 10. Known limitations

* ROI coordinates were measured on one modern-layout card.  Pendulum cards and
  very old layouts may need their own template (`backend/app/core/layout.py`).
* Orientation uses a text-density heuristic (0 vs 180 degrees) plus an OCR retry;
  extreme angles or heavy glare can still defeat detection.
* Rarity cannot be read from text; when several rarities share a set code the
  result lists them all as candidates.
* The dev catalog contains only a handful of cards; run the full sync before
  scanning arbitrary cards.
* CPU-only OCR takes roughly one to three seconds per photo.
* Data and images come from YGOPRODeck.  Review their terms (and Konami's) before
  any commercial use.  TCGplayer / Cardmarket integrations are intentionally not
  required and can be added behind `MarketPriceProvider`.

## 11. Future: DINOv2 visual fallback

`app/services/visual/` defines `VisualRecognizer` with `recognize(image) -> [VisualCandidate]`.
The pipeline only calls it when the OCR-based score is below
`VISUAL_TRIGGER_BELOW_CONFIDENCE`.  Plan: embed the normalized card (or artwork
crop) with DINOv2, store embeddings for all artworks (NumPy / FAISS), and feed the
nearest cards to the resolver as extra candidates.  Nothing in the pipeline has to
change; set `VISUAL_RECOGNIZER=dinov2` once implemented.

## 12. Migration paths

* SQLite -> PostgreSQL: set `DATABASE_URL`; the schema uses plain SQLAlchemy types.
* Local files -> object storage: `Artwork.local_path` is relative to `CARD_IMAGE_DIR`.
* Local OCR -> cloud fallback: implement `OCRProvider` (Google Vision / Azure) and
  select it via `OCR_PROVIDER`.
