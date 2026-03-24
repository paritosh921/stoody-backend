/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_EXAMPEN_API_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
