# PulseCheck AI — Showcase Kit

Copy-paste-ready content for LinkedIn, your resume, your portfolio site, and GitHub. Replace `<your-username>`, `<repo-name>`, and the live-demo URL once you've deployed (see the main README's "Deploying the demo on GitHub Pages" section).

---

## 1. LinkedIn — Project section / post

**Project title:** PulseCheck AI — Predictive Digital Wellness Platform

**Description (Project section, ~500 char limit):**
> Rebuilt a basic EDA notebook into a full-stack ML platform that predicts burnout/stress risk in real time from digital-habit check-ins. Trained a RandomForest (R²=0.97) + KMeans persona clustering, built a FastAPI backend and an offline rule-based guided chatbot, and added an optional LLM explanation layer. Caught and fixed a model blind spot that would have silently disabled a safety-critical crisis-escalation feature.

**Full LinkedIn post (feed post):**
```
🧠 Built PulseCheck AI — a predictive digital wellness platform.

I took an old EDA-only notebook (screen time, sleep, stress survey data)
and rebuilt it into a real applied Data Science / ML system:

→ Engineered a transparent, auditable 0–10 risk-scoring formula and
  trained a RandomForest regressor to validate it's learnable from raw
  check-in data (R² = 0.97, MAE = 0.18)

→ Used KMeans clustering to segment users into 4 interpretable "wellness
  personas" — named dynamically from real cluster centroids, not hardcoded

→ Found a real ML failure mode during testing: the trained model
  silently under-predicted extreme worst-case inputs it never saw in
  training (6.7 vs a true value of 8.58) — which would have suppressed a
  safety-critical crisis-escalation alert. Fixed it by using an
  auditable formula for that decision instead of a black-box prediction.

→ Added an optional LLM explanation layer (Groq/Llama, free tier) that narrates
  *why* a score is what it is in plain language — with a fully offline
  rule-based fallback, so the app never depends on an API key to run

→ Built an offline rule-based conversational check-in with a
  crisis-keyword safety net that surfaces real mental-health resources
  independent of the model flow

→ Shipped both a full-stack version (FastAPI + trained models) and a
  zero-dependency static demo — ported the actual trained model math
  into vanilla JS so anyone can try it instantly, no backend required

Stack: Python, scikit-learn, FastAPI, vanilla JS, optional Groq/Gemini API (free tier).

🔗 Live demo: https://<your-username>.github.io/<repo-name>/
🔗 Code: https://github.com/<your-username>/<repo-name>

#DataScience #MachineLearning #Python #FastAPI #Portfolio
```

---

## 2. Resume bullets

Pick 3–4 depending on space. Lead with the one matching the role you're applying for (ML-heavy vs. full-stack vs. product-minded).

```
PulseCheck AI — Predictive Digital Wellness Platform | Python, scikit-learn, FastAPI, JavaScript
• Rebuilt an exploratory-analysis notebook into a full-stack ML system: engineered features,
  trained a RandomForest risk model (R²=0.97) and a KMeans persona-clustering model, and served
  both through a FastAPI backend with a live web dashboard.
• Identified and resolved a model blind spot where the trained regressor under-predicted
  out-of-distribution extreme cases, which would have silently disabled a safety-critical
  escalation feature — replaced with an auditable, transparent scoring formula.
• Built an optional LLM-powered explanation layer (Groq/Gemini, free-tier providers) with a deterministic offline
  fallback, ensuring the application has zero hard dependency on external API availability.
• Designed and shipped a zero-backend static demo by porting trained model parameters
  (StandardScaler + KMeans centroids) directly into client-side JavaScript, enabling a
  free, live GitHub Pages deployment with no hosting cost.
• Implemented a rule-based conversational check-in flow with a crisis-keyword safety net that
  surfaces real mental-health resources independent of the primary prediction pipeline.
```

---

## 3. Portfolio website

**Suggested page structure:**

1. **Hero:** Project name + tagline + live demo button + GitHub button + hero screenshot (`screenshots/hero.png`)
2. **The problem → the rebuild:** 2–3 sentences — old notebook did EDA only, this closes the loop into a working system
3. **Architecture diagram:** copy the ASCII/diagram from the README, or recreate visually (data → features → models → API → UI)
4. **The interesting engineering decision** (this is the section that gets you interview questions — don't skip it):
   > "My RandomForest model scored 0.97 R² on held-out data, but testing edge cases revealed it silently under-predicted extreme inputs never seen during training. For the crisis-escalation logic — a safety-critical decision — I chose an auditable formula over the marginally-better black box. Explainability beat raw performance for this specific decision."
5. **Model card:** R² 0.97, MAE 0.18, 4 personas, 500-row dataset (be upfront about the size — it shows self-awareness).
6. **Screenshots gallery:** use all 5 images in `/screenshots` — check-in form, results with AI insight, chat widget, model metrics.
7. **Tech stack row:** logos/badges for Python, scikit-learn, FastAPI, JavaScript, Groq, Gemini.
8. **Links:** live demo, GitHub repo, one-page PDF (`PulseCheck_AI_One_Pager.pdf`) for a quick download/print version.

---

## 4. GitHub

**Repo description (short, under the repo name):**
```
Predictive digital-wellness platform: risk scoring, persona clustering, rule-based recommendations, and an optional LLM explanation layer. FastAPI + scikit-learn + vanilla JS. Live demo via GitHub Pages.
```

**Topics/tags to add:**
```
machine-learning  data-science  fastapi  scikit-learn  python  javascript
mental-health  nlp  llm  groq  gemini  portfolio-project  kmeans-clustering
```

**About section:**
- Website: your GitHub Pages demo URL
- Check "Use your GitHub Pages website" if prompted

**Pin it:** Settings → your profile → pin this repo so it's one of the first things visitors see.

---

## Quick checklist before you publish

- [ ] Replace all `<your-username>` / `<repo-name>` placeholders (README badges, this file, footer of the PDF)
- [ ] Push to GitHub, enable Pages on `/docs`, confirm the live demo link actually loads
- [ ] Update the one-pager PDF's footer links if you regenerate it (source: `one_pager.html`, rebuild with `wkhtmltopdf`)
- [ ] Add your name/contact to the portfolio page and resume bullets
- [ ] Optional: record a 30–60s screen capture of the check-in flow for the LinkedIn post — video posts get meaningfully more reach than static screenshots
