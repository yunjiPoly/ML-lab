/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the backend API. Empty string = same origin (dev proxy). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
