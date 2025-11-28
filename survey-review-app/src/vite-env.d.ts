/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_LINZ_BASEMAPS_API_KEY: string
    readonly VITE_DATA_API_URL: string
    // Add other environment variables here as needed
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}