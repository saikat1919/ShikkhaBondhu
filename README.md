# ShikkhaBondhu - PDF Question Answering

Upload one or more PDFs and ask questions about them. Built with:

- **Ingestion:** Markitdown + Camelot (tables)
- **Retrieval:** Hybrid search — ChromaDB (dense/semantic) + BM25 (sparse/keyword),
- **Generation:** Conversational, history-aware query rewriting + LLM answer generation with reasoning when needed
- **Interface:** Streamlit

Everything is processed in-memory per session — nothing is persisted once the
session ends.

## Setup (local development)
1. Create a python virtual environment
2. Activate the virtual environment
3. Clone the repo using `git clone https://github.com/saikat1919/ShikkhaBondhu.git`
4. Install the required packages by `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your actual API keys
6. Run the command `streamlit run app.py`

## Required environment variables

See `.env.example` and replace it with .env where you'll need:
- `HF_TOKEN`: Create a READ scope HF token from your Huggingface account.
- `GROQ_API_KEY`: Create a new API token from https://console.groq.com/keys

## 🌐 Live Demo
👉 https://shikkhabondhu-for-students.streamlit.app/

## 📄 Sample Test PDFs
For judges: sample documents used for testing → https://drive.google.com/drive/folders/1LUrVNoN2Gz-Rl7v3vGEBA9yj2qCPiKW1?usp=sharing

