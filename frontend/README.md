# Frontend React

Cette arborescence accueille la future interface React (par ex. Vite + TypeScript).

## Structure suggérée

- `src/app/` : pages et routing (React Router).
- `src/features/` : composants métier (ex: `verbatims`, `analysis`, `auth`).
- `src/shared/` : design system léger (boutons, layouts) et hooks communs.
- `src/api/` : clients typés générés à partir du schéma OpenAPI du backend.

## Démarrage (exemple Vite)

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm run dev
```

Connectez l'UI au backend FastAPI (ports et CORS à aligner). Les écrans peuvent réutiliser les modules Streamlit existants comme source fonctionnelle (tableaux marketing, IA rating, etc.).
