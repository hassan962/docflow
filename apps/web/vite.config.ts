import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs in Docker on :8000. Proxying /v1 through Vite keeps the
// browser on one origin, so the API needs no CORS rules.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://localhost:8000",
      "/healthz": "http://localhost:8000",
    },
  },
});
