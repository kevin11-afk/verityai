import type { Config } from "tailwindcss";

// Design tokens. Light, warm off-white canvas, white surfaces, deep navy ink,
// muted slate borders and restrained indigo / teal accents.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F7F5F0",
        surface: "#FFFFFF",
        sunken: "#F0EDE5",
        ink: { DEFAULT: "#0B1B33", 2: "#34425C", 3: "#5A6781" },
        line: { DEFAULT: "#E4E0D7", strong: "#CBC6B8" },
        brand: { DEFAULT: "#3F3CBB", soft: "#ECEBFA", ink: "#2E2B94" },
        teal: { DEFAULT: "#0F766E", soft: "#E3F2F0" },
        // Result semantics: deliberately not traffic-light neon.
        signal: {
          ai: "#A8243A", "ai-soft": "#FBECEE",
          real: "#0F766E", "real-soft": "#E3F2F0",
          unsure: "#8A5A00", "unsure-soft": "#FBF1DA",
        },
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ['"JetBrains Mono Variable"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 rgba(11,27,51,0.04), 0 1px 3px rgba(11,27,51,0.05)",
        pop: "0 12px 32px -8px rgba(11,27,51,0.22), 0 2px 6px rgba(11,27,51,0.08)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        indeterminate: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(350%)" } },
      },
      animation: { indeterminate: "indeterminate 1.4s ease-in-out infinite" },
    },
  },
  plugins: [],
} satisfies Config;
