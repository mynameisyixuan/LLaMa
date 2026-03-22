/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // 确保扫描到你的 App.jsx
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}