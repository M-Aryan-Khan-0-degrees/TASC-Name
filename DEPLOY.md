# Deploying the TASC Naming Engine

## 0. What's in this folder
- `app.py` — the whole app
- `requirements.txt` — dependencies
- `.streamlit/secrets.toml.example` — template, not real secrets
- `.gitignore` — keeps secrets and junk out of the repo

## 1. Add your brief file
Copy your `TASC_Naming_Brief_v1.docx` into this folder, right next to
`app.py`. The filename must match exactly (`BRIEF_PATH` in `app.py`) unless
you edit that constant.

## 2. Push to GitHub
```bash
cd tasc_naming_app
git init
git add .
git commit -m "TASC naming engine"
```
Create a new **private** repo on GitHub (private keeps your brief and any
naming logic off the public internet — the app itself will still be public,
just not the source), then:
```bash
git remote add origin https://github.com/<you>/tasc-naming-engine.git
git branch -M main
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app", pick your repo, branch `main`, main file `app.py`.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key"
   ```
   (Add `APP_PASSWORD = "..."` too if you want a login gate — see below.)
4. Click Deploy. First build takes a few minutes (installing deps).

You'll get a URL like `https://tasc-naming-engine.streamlit.app` — that's
what you share with anyone who should use it.

## 4. Protecting your API budget
Since the app uses **your** Anthropic key, anyone with the link can
generate/verify names at your expense. Two ways to limit that:
- **Password gate (built in):** set `APP_PASSWORD` in secrets. Visitors see
  a password box before anything else. Share the password only with people
  who should have access.
- **Spending limits:** set a monthly budget/alert on your Anthropic
  Console (https://console.anthropic.com) regardless of the password, as a
  backstop.

## 5. Known limitations to expect
- **DuckDuckGo search (`ddgs`)** can occasionally rate-limit or block
  requests coming from cloud data-center IPs (this is a DDG-side behavior,
  not something in your code). If verification search results start coming
  back empty/failed for everyone, this is the most likely cause — there's
  no reliable fix beyond retrying later or swapping in a paid search API.
- **Streamlit Community Cloud free tier** has modest RAM. The app is built
  to skip `sentence-transformers` entirely (SIMPLE_MODE) as long as your
  brief is under ~12,000 characters, which should be true for most naming
  briefs. If your brief is longer and the app falls back to the embedding
  model, watch the app logs for out-of-memory restarts — if that happens,
  either trim the brief or upgrade to a paid Streamlit/HF tier.
- **Concurrent users** share the same cached brief index (fast, no
  duplicate work) but each get separate name pools (`st.session_state` is
  per-browser-session, so one visitor's generated names won't show up for
  another).

## 6. Updating later
Any `git push` to `main` auto-redeploys. Editing secrets in the dashboard
does not require a push.
