/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "pen-pending": "#94a3b8",
        "pen-connecting": "#facc15",
        "pen-syncing": "#60a5fa",
        "pen-complete": "#4ade80",
        "pen-failed": "#f87171",
        "pen-timeout": "#fb923c",
        "dongle-healthy": "#22c55e",
        "dongle-degraded": "#eab308",
        "dongle-failed": "#ef4444",
      },
    },
  },
  plugins: [],
};
