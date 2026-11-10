# MyCustomRAG

A Streamlit app that combines a custom chatbot with document RAG. Upload PDF, DOCX, TXT, JSON, MD or CSV files, then ask questions. Answers in document mode come only from your files and show the source chunks. You can also switch to general chat.

This project joins ideas from hasnaaain-GPT-4.2 (chat UI, API key gate, export) and the course RAG app (FAISS retrieval, maroon and cream UI, citations).

## Features

* OpenAI API key checked on a welcome screen before chat starts
* Two modes: Documents (RAG) and General chat
* Multi-file upload with size and count limits
* FAISS vector index with adjustable chunk size, overlap and top-k
* Streaming answers with source cards in RAG mode
* Personality presets and optional system role for general chat
* Chat history in the sidebar (new, switch, delete)
* Export chat as markdown or JSON; import a JSON chat back
* Maroon and cream theme with Fraunces and DM Sans

## Security and deployment choices

These choices matter on Streamlit Community Cloud, where many people share one app process:

* Chat history and the FAISS index live only in each visitor's session. Nothing is written to a shared folder, so visitors cannot see each other's chats.
* The API key stays in the session. It is never saved to disk.
* Filenames are sanitised before display to avoid HTML injection.
* Uploads are capped (8 files, 15 MB each, character and chunk limits).
* FAISS indexes are built in memory from the current upload. The app does not load pickle indexes from user files.
* `.env` and secrets are listed in `.gitignore`.

History is lost when the browser session ends unless the visitor exports the chat. That is intentional for a public deploy.

## Requirements

* Python 3.10 or newer
* An OpenAI API key

## Setup

```
cd "E:\Corvit Ai course\week 7\sun\MyCustomRAG"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: create a `.env` file with `OPENAI_API_KEY=your_key` for local use. Do not commit that file.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (include `app.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml` and `.gitignore`).
2. At https://share.streamlit.io choose that repo and set the main file to `app.py`.
3. Each visitor enters their own OpenAI key in the app.

Do not put your personal API key in the repo or in Streamlit secrets unless you want every visitor to use your key and your billing.

## Files

| File | Purpose |
|---|---|
| `app.py` | Full application |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Maroon and cream Streamlit theme |
| `.gitignore` | Keeps secrets and local caches out of git |

## Notes

* Document mode uses `text-embedding-3-small` for embeddings and the chat model you pick in Settings.
* General chat does not use the vector index.
* If you change chunk size or overlap, the app rebuilds the index for the current upload.
