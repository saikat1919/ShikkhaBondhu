# InfoBee — PDF Question Answering (RAG)

Upload one or more PDFs and ask questions about them. InfoBee answers **only**
from the content of the uploaded documents — if the answer is not in the
documents, it says so instead of guessing.

🌐 **Live demo:** https://infobee.streamlit.app/

## How it works

- **Ingestion:** Markitdown + Camelot (tables)
- **Retrieval:** Hybrid search — ChromaDB (dense/semantic) + BM25 (sparse/keyword),
  combined via weighted Reciprocal Rank Fusion
- **Generation:** History-aware query rewriting + grounded, context-only answer generation
- **Interface:** Streamlit

Everything is processed in-memory per session — nothing is persisted once the
session ends.

## Run it locally

```bash
git clone https://github.com/saikat1919/InfoBee.git
cd InfoBee
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
streamlit run app.py
```

### API keys

Put these in `.env` (see `.env.example`):

| Variable | Where to get it |
| --- | --- |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `HF_TOKEN` | A READ-scope token from your Hugging Face account settings |

That is all the setup needed: no `secrets.toml`, no Docker. On Streamlit Cloud
the same two keys go in the app's **Secrets** panel instead; the app reads from
either source.
