import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Built files go to ../product/static, which the Python server mounts at /product.
// base "./" keeps asset URLs relative; index.html injects a <base> tag at runtime so
// deep links and hosts that add a path prefix (e.g. Streamlit Cloud) both resolve.
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  build: {
    outDir: "../product/static",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 700,
  },
  server: {
    port: 5173,
    // Local development: proxy API calls to a locally running `uvicorn api.main:app --port 8000`.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
