# MediBot frontend

A minimal Next.js interface for the existing Milestone 7 FastAPI API.

## Configuration

Copy `.env.example` to `.env.local`. `NEXT_PUBLIC_API_BASE_URL` is the browser-visible base URL of FastAPI and defaults in the client to `http://localhost:8000` when omitted.

The backend allows `http://localhost:3000` by default. Set `MEDIBOT_FRONTEND_ORIGIN` in the backend `.env` if the frontend uses another origin.

## Commands

```bash
npm install
npm run dev
npm test
npm run lint
npm run build
```

The bearer token is held in `sessionStorage`, scoped to the current browser tab, and cleared on logout or an authenticated `401` response. The frontend never decodes it or makes permission decisions.
