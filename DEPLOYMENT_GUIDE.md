# Deployment Guide — From Zero to Live

This walks through everything: how the frontend and backend are already connected, what GitHub Actions does and doesn't do here, how to get the repo on GitHub, and two deployment paths depending on what you want live.

---

## 1. How the frontend and backend are already connected

You don't need to "wire them up" — they're already connected, and it's worth understanding how, because it explains every deployment decision below.

- `backend/main.py` is a FastAPI app. One of its routes, `GET /`, returns `frontend/index.html` directly (`FileResponse`).
- The same app also exposes `POST /api/predict`, `POST /api/chat/start`, `POST /api/chat/message`.
- Inside `frontend/index.html`, the JavaScript does:
  ```js
  const API_BASE = window.location.origin;
  fetch(`${API_BASE}/api/predict`, { method: 'POST', ... })
  ```
  `window.location.origin` means "whatever host and port served this page." Since FastAPI serves both the page and the API from the same process, this is automatically correct wherever you run it — `localhost:8000` locally, or your real domain once deployed. **There's no separate frontend server, no CORS config needed, and no URL to hardcode.**

This is why running the whole app locally is one command:
```bash
pip install -r requirements.txt
python src/train_models.py
uvicorn backend.main:app --reload --port 8000
```
Open `http://localhost:8000` — frontend and backend are already talking to each other.

The **static demo** in `/docs` is a different, deliberately disconnected build: it has no `fetch()` calls at all. The risk formula and persona-classifier math are reimplemented in plain JavaScript and run entirely in the browser (see `docs/README.md`). That's *why* it can go on GitHub Pages, which only serves files and cannot run Python.

---

## 2. What GitHub Actions does and doesn't do here

This trips people up, so to be explicit:

- **GitHub Actions can NOT host your live backend.** Actions runners spin up, run a job (e.g. "install deps, run tests"), and then shut down completely. There is no persistent process left running afterward for the public to hit. You cannot deploy a FastAPI server "onto" Actions and have people use it.
- **GitHub Actions CAN run automated checks** on every push — this repo includes `.github/workflows/ci.yml`, which installs dependencies, retrains the models, boots the server, and fires real requests at it (including a check that the crisis-escalation logic actually fires for a critical check-in). This is a correctness safety net, not a deployment mechanism.
- **GitHub Pages does NOT use Actions by default** in the setup below — you just point it at a folder (`/docs`) on a branch, and GitHub's own infrastructure serves those static files directly. No workflow required.

So: two genuinely different things—

| Thing | What it needs | What lives there |
|---|---|---|
| Static demo (`/docs`) | GitHub Pages (free, built-in) | HTML/CSS/JS only — no Python, no backend |
| Full-stack app (`/backend`, `/src`) | A real Python host (Render, Railway, Fly.io, etc.) | The actual FastAPI server, trained models, API |

You can do just the first one, just the second, or both. Most portfolio use cases only need the first.

---

## 3. Get the repo onto GitHub (do this first, either way)

1. Create a new empty repository on GitHub (no README/gitignore/license — you already have those). Note the URL, e.g. `https://github.com/<you>/pulsecheck-ai.git`.
2. In the unzipped project folder:
   ```bash
   cd pulsecheck-ai
   git init
   git add .
   git commit -m "Initial commit: PulseCheck AI"
   git branch -M main
   git remote add origin https://github.com/<you>/pulsecheck-ai.git
   git push -u origin main
   ```
3. Refresh the GitHub page — you should see all the files, and (after a minute) a green check next to your commit once `ci.yml` finishes running automatically.
4. If the CI check turns red, click it to see which step failed — the logs show exactly which command failed and why.

---

## 4. Deploy the static demo (GitHub Pages) — do this

This gets you a free, permanent, shareable link with zero ongoing cost or maintenance.

1. On GitHub, go to your repo → **Settings** → **Pages** (left sidebar).
2. Under **Build and deployment**, set **Source: Deploy from a branch**.
3. Set **Branch: `main`**, folder: **`/docs`**. Click **Save**.
4. Wait 1-2 minutes, then refresh the Pages settings page — it'll show "Your site is live at `https://<you>.github.io/pulsecheck-ai/`".
5. Open that URL. You should see the full PulseCheck UI, and the check-in / chat should work immediately (it's all client-side).
6. Go back and update the placeholder links (`<your-username>`) in `README.md`, `SHOWCASE_KIT.md`, and the PDF footer with this real URL, then commit and push the change.

This is the link to put on LinkedIn, your resume, and your portfolio site.

---

## 5. (Optional) Deploy the full backend live

Do this only if you specifically want the *real* trained RandomForest model, the live chat session state, and the LLM explanation layer running on a public URL — not required for the demo link above.

GitHub Pages can't run this (see section 2), so you need an actual Python host. **Render** has a straightforward free tier and native GitHub integration, so it's a reasonable default:

1. Go to [render.com](https://render.com) and sign up (GitHub login is fine).
2. **New +** → **Web Service** → connect your GitHub account → select the `pulsecheck-ai` repo.
3. Configure:
   - **Name:** `pulsecheck-ai` (or anything)
   - **Runtime:** Python 3
   - **Build Command:**
     ```
     pip install -r requirements.txt && python src/train_models.py
     ```
   - **Start Command:**
     ```
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance type:** Free
4. (Optional) Add environment variables if you want the real LLM explanation layer instead of the rule-based fallback:
   - `GROQ_API_KEY` — free key from [console.groq.com](https://console.groq.com)
   - `GEMINI_API_KEY` — free key from [aistudio.google.com](https://aistudio.google.com)
   - Neither is required — without them, `ai_explanation.source` will just report `"rule_based"` instead of `"llm"`, and everything else works identically.
5. Click **Create Web Service**. First deploy takes a few minutes (installing deps + training models). Watch the logs — you're looking for `Uvicorn running on http://0.0.0.0:...`.
6. Render gives you a public URL like `https://pulsecheck-ai.onrender.com`. Open it — this serves the *full* app (frontend + real backend), same as running it locally.
7. Every future `git push` to `main` auto-redeploys (Render watches the repo by default).

**Free-tier caveat:** Render's free web services spin down after inactivity and take ~30-50 seconds to wake up on the next request. Fine for a portfolio demo people click occasionally; not meant for constant traffic. If that matters, Railway and Fly.io are comparable alternatives with similar free tiers and the same Build/Start command pattern above.

---

## 6. Full checklist, start to finish

- [ ] Unzip the project locally, confirm it runs (`uvicorn backend.main:app --reload`)
- [ ] Create the GitHub repo, `git push` (section 3)
- [ ] Confirm the CI check goes green on GitHub
- [ ] Enable GitHub Pages on `/docs` (section 4) — this is your primary shareable link
- [ ] Replace `<your-username>` placeholders everywhere with the real Pages URL
- [ ] (Optional) Deploy the live backend on Render with a free `GROQ_API_KEY` or `GEMINI_API_KEY` (section 5)
- [ ] Add both links (and the repo link) to LinkedIn / resume / portfolio using `SHOWCASE_KIT.md`
