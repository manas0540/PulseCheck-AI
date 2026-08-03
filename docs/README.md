# PulseCheck AI — Static Demo Build

A **fully self-contained, single-file** version of PulseCheck AI — no backend, no server, no API keys, no build step. Everything runs in the browser.

This exists so the project has a **live, clickable demo link** (GitHub Pages / Netlify / Vercel) for a portfolio, resume, or LinkedIn post, without needing to host a Python backend anywhere.

> The full-stack version — FastAPI backend, trained models retrainable from raw data, optional LLM explanation layer — lives in the main repo. This static build is a faithful port of its output, not a simplified mock.

## What's actually ported here (not faked)

- **Risk score** — the exact same weighted formula as `src/data_processing.py::compute_risk_score`.
- **Persona classification** — the *real* trained `StandardScaler` mean/scale and `KMeans` cluster centers from the full-stack repo, hardcoded as constants and run as a nearest-centroid classifier in JavaScript. This is genuinely the trained model's decision boundary, not an approximation.
- **Recommendation + escalation logic** — same rule-based tips and the same 8.5/10 crisis-escalation threshold, with the same real crisis resources.
- **AI insight explanation** — the same rule-based natural-language explainer (the offline fallback path from the full-stack repo's optional LLM layer).
- **Guided chat** — the same conversational state machine, running entirely client-side.

Verified against the live FastAPI backend with matching test cases (see the note at the bottom) before publishing.

## Run it

Just open `index.html` in a browser. That's it — no `npm install`, no `pip install`, nothing to start.

## Deploying

This folder is already set up to be published via **GitHub Pages** directly from the main repo (`Settings → Pages → Deploy from branch → main → /docs`) — see the main [README](../README.md#deploying-the-demo-on-github-pages) for the exact steps. You can also drag-and-drop this folder into Netlify or Vercel for the same result.

## Why a separate static build

Retraining the RandomForest regressor client-side isn't practical (300 trees, ~3MB of serialized weights) — so this build intentionally uses the same transparent risk-score formula the full-stack app already treats as its source of truth for the served score (see the full repo's README for why: the formula is safety-critical and auditable, while the RF model is kept as a secondary validation signal). The persona clustering, being a handful of numbers, ports over exactly.

## License

MIT — see the main repo.
