# Deploying the public demo on Streamlit Community Cloud

Streamlit Community Cloud hosts public Streamlit apps for free, straight from a GitHub repository. RepoLens has a **public demo mode** built for it:

- the model provider is fixed by the server (Groq) and the API key is **never sent to visitors' browsers**,
- each visitor session gets a limited number of AI answers (default 8) and the agent's tool budget is capped at 4 steps,
- repository size limits are tighter (2,500 files / 25 MB of source),
- keyword (BM25) search is used, because Ollama embeddings are not available on free hosting.

## Costs: none

- Streamlit Community Cloud is free for public apps.
- Groq's **free tier cannot charge you**: billing only exists on Groq accounts that add a payment method and upgrade. When the free quota (8,000 tokens/minute, 1,000 requests/day per model) runs out, Groq rejects requests with a 429 error until the window resets; RepoLens shows "quota reached" instead of retrying. **Do not add a payment method to your Groq account** and nothing can ever be billed.
- Worst case if the demo gets busy: it stops answering AI questions until the quota resets. Search-only mode keeps working.

## Steps (about 5 minutes)

1. Make sure the repository is on GitHub and public.
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **Create app** → **Deploy a public app from GitHub**, then fill in:
   - Repository: `Max0786234/RepoLens`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a name, e.g. `repolens`
4. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets — paste (with your own key):

     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     REPOLENS_PROVIDER = "groq"
     REPOLENS_PUBLIC_DEMO = "1"
     REPOLENS_EMBED_BACKEND = "none"
     REPOLENS_MAX_QUESTIONS = "8"
     ```
5. Click **Deploy**. The first start installs `requirements.txt` (numpy + streamlit) and takes a minute or two.
6. Copy the app URL into the README's "Live demo" line.

## Changing settings later

App → **⋮ → Settings → Secrets** on share.streamlit.io. Useful knobs:

| Secret | Default | Meaning |
|---|---|---|
| `REPOLENS_MAX_QUESTIONS` | `8` | AI answers per visitor session (0 = unlimited) |
| `REPOLENS_MAX_STEPS` | `6` (capped at 4 in public mode) | agent tool budget per question |
| `REPOLENS_MAX_CONTEXT_TOKENS` | `5500` | largest request sent to Groq |
| `REPOLENS_LLM_MODEL` | `openai/gpt-oss-120b` | e.g. `openai/gpt-oss-20b` for a separate quota |

## Rotating the key

If the key was ever shared (for example pasted into a chat), create a new one at https://console.groq.com/keys, delete the old one, and update the secret above (and your local `.env`).
