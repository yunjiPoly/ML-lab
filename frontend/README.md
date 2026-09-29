# Yu-Gi-Oh! card scanner - web frontend

Mobile-first React + TypeScript + Vite UI for the card recognition API. Take a
photo with the phone camera (or upload an image), send it to `POST /api/recognize`
and see the recognized card, its set code and the exact printing.

Plain React with two small CSS files worth of styling (CSS variables, light and
dark mode); no state library, no UI kit, no Tailwind.

## Requirements

- Node 20.19+ (developed with Node 24 / npm 11)
- The FastAPI backend running on `http://localhost:8000`
  (from `backend/`: `uvicorn app.main:app --host 0.0.0.0 --port 8000`)

## Run

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (also reachable on the LAN, see below)
```

| Script              | What it does                                                   |
| ------------------- | -------------------------------------------------------------- |
| `npm run dev`       | Vite dev server on port 5173, `/api` proxied to the backend    |
| `npm run typecheck` | `tsc --noEmit` for `src/` and for `vite.config.ts`             |
| `npm run build`     | Type-check, then production build into `dist/`                 |
| `npm run preview`   | Serve `dist/` on port 4173 (same `/api` proxy)                 |

## Open the scanner from a phone

1. Start the backend with `--host 0.0.0.0` so it listens on the LAN, and start
   `npm run dev` (Vite binds to all interfaces: `server.host = true`).
2. Find the PC's LAN address: `ipconfig` on Windows (the "IPv4 Address" of your
   Wi-Fi adapter, e.g. `192.168.1.20`). Allow Node through Windows Firewall on
   private networks if it prompts.
3. On the phone (same Wi-Fi), open `http://192.168.1.20:5173`.
4. **Camera on the phone needs HTTPS.** Browsers only expose `getUserMedia` in a
   secure context: `http://localhost` qualifies on the PC, a plain-HTTP LAN
   address does not. Start the dev server with a self-signed certificate:

   ```powershell
   $env:VITE_HTTPS = "true"; npm run dev      # PowerShell
   ```

   ```bash
   VITE_HTTPS=true npm run dev                # bash
   ```

   or put `VITE_HTTPS=true` in `frontend/.env.local`. Then open
   `https://192.168.1.20:5173` on the phone and accept the certificate warning
   once. The `/api` proxy still talks plain HTTP to the backend, so nothing
   changes on the backend side.

   Without HTTPS the **Upload image** mode still works everywhere: its file
   input uses `capture="environment"`, which opens the phone's camera app.

The backend's CORS defaults (`CORS_ORIGINS` / `CORS_ORIGIN_REGEX` in the root
`.env.example`) already allow localhost and private LAN origins. With the dev
proxy the browser only talks to the Vite origin, so CORS is not involved at all.

## Configuration

Copy `.env.example` to `.env.local` (git-ignored) and adjust:

| Variable                | Default                 | Meaning                                                                |
| ----------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `VITE_API_BASE_URL`     | *(empty)*               | API origin used by the browser. Empty = same origin (dev proxy).       |
| `VITE_API_PROXY_TARGET` | `http://localhost:8000` | Where the dev/preview proxy forwards `/api`.                           |
| `VITE_HTTPS`            | `false`                 | `true` serves the dev server over HTTPS (self-signed) for phone camera |

Shell environment variables take precedence over `.env*` files.

## API contract used by the UI

| Endpoint                             | Client function (`src/api/client.ts`) |
| ------------------------------------ | ------------------------------------- |
| `POST /api/recognize` (multipart, field `image`, file name `capture.jpg`; `?debug=true` when the developer option is ticked) | `recognizeImage(blob, { debug })` |
| `GET /api/health`                    | `getHealth()`                         |
| `GET /api/search?q=`                 | `searchCards(q)`                      |
| `GET /api/cards/{id}`                | `getCard(id)`                         |

Non-2xx responses throw an `ApiError` carrying the server's `detail` message;
network failures throw an `ApiError` with `status === 0`.

`src/types/recognition.ts` mirrors `backend/app/schemas/recognition.py`
(`RecognitionResult`, `RecognitionStatus`, `OcrFieldResult`, `CardSummary`,
`PrintingSummary`, `Candidate`, `DetectionInfo`). Keep the two in sync.

## Project layout

```
frontend/
  index.html                 viewport meta, favicon, #root
  vite.config.ts             host=true, port 5173, /api proxy, optional basicSsl()
  src/
    main.tsx                 entry point
    App.tsx                  shell: header + health, scanner / result screens
    styles.css               CSS variables, light/dark, layout, components
    api/client.ts            fetch wrapper + ApiError
    types/recognition.ts     backend JSON contract
    types/cards.ts           card / search / health types
    hooks/useCamera.ts       getUserMedia lifecycle + friendly errors
    utils/format.ts          percent / ms / bytes formatting
    components/
      ScannerScreen.tsx      camera / upload mode switch, busy overlay
      CameraCapture.tsx      live preview, capture to JPEG (q=0.92), retake / use photo
      CardGuideOverlay.tsx   59:86 card guide over the preview
      UploadPicker.tsx       file input (capture=environment) + drop zone + preview
      ResultCard.tsx         card, set code, rarity, status badge, notes, candidates
      CandidateList.tsx      candidates (primary styling when AMBIGUOUS)
      StatusBadge.tsx        status -> label / colour
      DeveloperDetails.tsx   collapsed OCR / detection / timing / raw JSON panel
      HealthIndicator.tsx    /api/health dot in the header
      ErrorAlert.tsx         inline API error with Retry
      ErrorBoundary.tsx      render-error fallback
      Spinner.tsx
```

## Privacy

Captured or uploaded images live only in memory (an object URL for the preview
and thumbnail) and are sent to the recognition API once. Nothing is written to
`localStorage`, cookies or IndexedDB.

## Production build

`npm run build` writes a static site to `dist/`. Serve it from any static host.
If the API lives on another origin, build with `VITE_API_BASE_URL=https://api.example.com`
and make sure the backend's CORS settings allow the site's origin.
