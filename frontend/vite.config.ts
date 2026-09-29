import { defineConfig, loadEnv, type PluginOption, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

/**
 * Vite configuration.
 *
 * - `server.host = true` binds to every interface so a phone on the same LAN can
 *   open the dev server at http(s)://<pc-lan-ip>:5173.
 * - `/api` is proxied to the FastAPI backend, so the browser only ever talks to
 *   the Vite origin (no CORS during development).
 * - `VITE_HTTPS=true` serves the dev server over HTTPS with a self-signed
 *   certificate. `getUserMedia` (the camera) is only available in a secure
 *   context, which browsers grant to `localhost` but not to a plain-HTTP LAN IP.
 *
 * Environment values are read from the shell first, then from `.env*` files.
 */
export default defineConfig(({ mode }) => {
  const fileEnv = loadEnv(mode, process.cwd(), "VITE_");
  const env = (key: string): string | undefined => process.env[key] ?? fileEnv[key];

  const useHttps = env("VITE_HTTPS") === "true";
  const proxyTarget = env("VITE_API_PROXY_TARGET") || "http://localhost:8000";

  const apiProxy: Record<string, ProxyOptions> = {
    "/api": {
      target: proxyTarget,
      changeOrigin: true,
    },
  };

  const plugins: PluginOption[] = [react()];
  if (useHttps) {
    plugins.push(basicSsl());
  }

  return {
    plugins,
    server: {
      host: true,
      port: 5173,
      proxy: apiProxy,
    },
    preview: {
      host: true,
      port: 4173,
      proxy: apiProxy,
    },
    build: {
      sourcemap: false,
      target: "es2022",
    },
  };
});
