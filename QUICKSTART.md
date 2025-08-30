# Ultra‑Fast Build Checklist

1. **Install Ollama**
   - `curl -fsSL https://ollama.com/install.sh | sh` (Linux/Mac)
   - Download MSI for Windows from the site.
   - Pull models: `ollama pull llama3:8b` and `ollama pull nomic-embed-text`

2. **Clone this folder to your machine** and `cd` into it.

3. **Create venv + install deps**: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

4. **Add PDFs to `data/`**, then `python ingest.py`

5. **Run UI**: `streamlit run app.py`

6. **(Optional) Langflow**: `uv pip install langflow -U && uv run langflow run` and wire **Ollama + Chroma** to reuse `.chroma/`.

7. **Done.** Everything runs locally. Switch models anytime using the sidebar or env var `CHAT_MODEL`.
