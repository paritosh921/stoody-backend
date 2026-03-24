/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f5ff',
          100: '#e0ebff',
          500: '#3b6cf5',
          600: '#2a56d4',
          700: '#1e42b0',
        },
      },
    },
  },
  plugins: [],
};
