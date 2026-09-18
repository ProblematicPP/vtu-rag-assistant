import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In the container nginx proxies /api and /health to the API, so the app always
// talks to its own origin. `npm run dev` needs the same shape locally, pointed
// at the API the compose stack publishes. (127.0.0.1, not localhost: that
// resolves to IPv6 here and never reaches the container.)
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: api, changeOrigin: true },
      "/health": { target: api, changeOrigin: true },
    },
  },
});
