"""
MyCustomRAG — custom chat + document RAG in one Streamlit app.

Combines the hasnaaain-GPT-4.2 chat UI with the RAG pipeline from the
course project. Theme: maroon and cream (Fraunces + DM Sans).

Deployment notes:
  - Chat history and the FAISS index live only in session_state.
    Nothing is written to a shared disk folder, so visitors cannot see
    each other's chats on Streamlit Community Cloud.
  - API keys stay in the browser session and are never saved to disk.
  - Users can export / import a chat as JSON to keep their own history.
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(find_dotenv(), override=True)


# Config
# --------------------------------------------------------------------------- #
BRAND = "MyCustomRAG"
MAX_FILE_BYTES = 15 * 1024 * 1024   # 15 MB per file
MAX_FILES = 8
MAX_TOTAL_CHARS = 800_000
MAX_CHUNKS = 2_500
MAX_HISTORY_MSGS = 24
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
ALLOWED_EXT = {"pdf", "docx", "txt", "json", "md", "csv"}

TYPE_STYLE = {
    "pdf": ("PDF", "#800020"),
    "docx": ("DOCX", "#6A2E35"),
    "txt": ("TXT", "#A45A2A"),
    "md": ("MD", "#6B4F3A"),
    "csv": ("CSV", "#4A6B41"),
    "json": ("JSON", "#4A6B41"),
}

PERSONAS = {
    "Helpful assistant": "You are a helpful, clear assistant. Be concise.",
    "Python tutor": "You are a patient Python tutor. Explain step by step with small examples.",
    "Code reviewer": "You are a senior engineer. Review for bugs, style and performance.",
    "Study buddy": "You are a study buddy. Summarise, quiz, and explain simply.",
    "Technical writer": "You are a technical writer. Prefer clear structure and plain language.",
}

st.set_page_config(
    page_title=BRAND,
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme (maroon + cream, Fraunces + DM Sans) — no emoji in the UI
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --cream: #FDFBF7;
    --sand: #F4ECE1;
    --ink: #2C1A1D;
    --muted: #7A696C;
    --maroon: #800020;
    --maroon-deep: #500013;
    --rose: #B03052;
    --gold: #D4A373;
    --line: #E8DCCF;
}
:root, .stApp { color-scheme: light; }
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--cream); }
header[data-testid="stHeader"] { visibility: visible; background: transparent; }
#MainMenu, footer { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 1100px; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes floatA { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-28px,16px) scale(1.1); } }
@keyframes floatB { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(24px,-14px) scale(1.08); } }
@keyframes dots { 0%,80%,100% { transform: scale(.5); opacity: .35; } 40% { transform: scale(1); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

section[data-testid="stSidebar"] { background: var(--sand); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: var(--ink); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif; color: var(--maroon-deep);
}
.side-title { font-family: 'Fraunces', serif; font-weight: 700; color: var(--maroon-deep);
              font-size: 1.15rem; margin: 1.1rem 0 .4rem 0; }
.side-label { font-size: .72rem; letter-spacing: .16em; font-weight: 600; color: var(--maroon);
              margin: 1rem 0 .45rem; text-transform: uppercase; }

[data-testid="stWidgetLabel"] p, label p, label { color: var(--ink) !important; font-weight: 500; }
div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: #fff !important; border-radius: 12px !important;
}
div[data-baseweb="input"] { border: 1px solid #D9C8B8 !important; }
div[data-baseweb="input"]:focus-within { border-color: var(--maroon) !important; box-shadow: 0 0 0 3px rgba(128,0,32,.12); }
input, textarea {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important;
    caret-color: var(--maroon) !important; background: transparent !important;
}
input::placeholder, textarea::placeholder { color: #9A878A !important; -webkit-text-fill-color: #9A878A !important; }
div[data-baseweb="select"] > div {
    background: #fff !important; border: 1px solid #D9C8B8 !important; border-radius: 12px !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
div[data-baseweb="select"] svg { fill: var(--maroon-deep) !important; }

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    background: #fff !important; color: var(--maroon-deep) !important;
    border: 1.5px solid var(--maroon) !important; border-radius: 12px; font-weight: 600;
    min-height: 42px; transition: transform .15s ease, background .2s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--maroon) !important; color: #fff !important;
    transform: translateY(-1px); box-shadow: 0 6px 14px rgba(128,0,32,.22);
}
.stButton > button p, .stDownloadButton > button p { color: inherit !important; }
.stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"],
.stApp [data-testid="stFormSubmitButton"] > button {
    background: var(--maroon) !important; color: #fff !important; border-color: var(--maroon) !important;
}
.stApp button[kind="primary"] p, .stApp button[data-testid="stBaseButton-primary"] p,
.stApp [data-testid="stFormSubmitButton"] > button p { color: #fff !important; }
[class*="st-key-chat_on_"] button { border-color: var(--maroon) !important; background: #FAF2F4 !important; }

.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, #420A14 0%, var(--maroon-deep) 50%, var(--maroon) 100%);
    border-radius: 24px; padding: 2.4rem 2.6rem; margin-bottom: 1.4rem;
    box-shadow: 0 12px 35px rgba(80,0,19,.22);
    border: 1px solid rgba(255,255,255,.1);
    animation: fadeUp .6s ease both;
}
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%; filter: blur(12px); pointer-events: none; }
.hero::before { width: 280px; height: 280px; right: -70px; top: -100px;
                background: radial-gradient(circle, rgba(212,163,115,.35), transparent 65%);
                animation: floatA 9s ease-in-out infinite; }
.hero::after  { width: 320px; height: 320px; right: 110px; bottom: -180px;
                background: radial-gradient(circle, rgba(176,48,82,.4), transparent 65%);
                animation: floatB 11s ease-in-out infinite; }
.hero > * { position: relative; z-index: 1; }
.hero h1 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.4rem;
           margin: 0 0 .4rem 0; color: #FFF8F0 !important; letter-spacing: -.5px; }
.hero p { margin: 0; font-size: 1.02rem; color: #FFF8F0; opacity: .92; max-width: 640px; line-height: 1.5; }
.hero .pills { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.hero .pill { font-size: .72rem; font-weight: 600; letter-spacing: .5px; color: #FFF8F0;
              padding: .28rem .75rem; border-radius: 999px;
              background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25); }

.stat { background: #fff; border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem 1.2rem; border-top: 4px solid var(--maroon);
        box-shadow: 0 4px 12px rgba(44,26,29,.03); animation: fadeUp .55s ease both; }
.stat.rose { border-top-color: var(--rose); }
.stat.gold { border-top-color: var(--gold); }
.stat .num { font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 700; line-height: 1.1; color: var(--ink); }
.stat .lbl { color: var(--muted); font-size: .82rem; margin-top: .2rem; }

.filerow { display: flex; align-items: center; gap: .75rem; background: #fff;
           border: 1px solid var(--line); border-radius: 12px; padding: .55rem .85rem; margin-bottom: .4rem; }
.badge { color: #fff !important; font-size: .66rem; font-weight: 700; letter-spacing: .7px;
         padding: .2rem .48rem; border-radius: 6px; min-width: 46px; text-align: center; }
.filerow .name { font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.filerow .meta { color: var(--muted); font-size: .8rem; }

.src { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--gold);
       border-radius: 12px; padding: .75rem 1rem; margin-bottom: .55rem; font-size: .88rem; }
.src .tag { font-weight: 600; color: var(--maroon-deep); font-size: .8rem; margin-bottom: .25rem; }

[data-testid="stFileUploader"] section { background: #fff !important; border: 2px dashed #D2BFB0; border-radius: 18px; }
[data-testid="stChatMessage"] { background: #fff !important; border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background: #fff !important; border-radius: 18px !important; border: 1.5px solid #D9C8B8;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--maroon); box-shadow: 0 6px 22px rgba(128,0,32,.15); }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 14px; }
[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 1.3rem; }

.typing { display: inline-flex; gap: 6px; padding: .35rem 0; }
.typing span { width: 9px; height: 9px; border-radius: 50%; background: var(--maroon); animation: dots 1.2s infinite ease-in-out; }
.typing span:nth-child(2) { animation-delay: .15s; background: var(--rose); }
.typing span:nth-child(3) { animation-delay: .3s; background: var(--gold); }

.empty { text-align: center; color: var(--muted); padding: 2rem 1.2rem;
         border: 1px dashed #D9C8B8; border-radius: 18px; background: rgba(255,255,255,.65); }
.empty b { color: var(--maroon-deep); }
.mode-chip { display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px;
             border-radius: 999px; font-size: .78rem; font-weight: 600; background: #fff;
             border: 1px solid var(--line); color: var(--maroon); }
.gate-note { color: var(--muted); font-size: .85rem; text-align: center; margin-top: .8rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
TYPING_HTML = '<div class="typing"><span></span><span></span><span></span></div>'

# --------------------------------------------------------------------------- #
# Session defaults (everything stays in-session for multi-user safety)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "api_key": "",
    "chats": {},
    "current": None,
    "vectorstore": None,
    "index_owner": None,   # chat id that owns the in-memory FAISS index
    "files_meta": [],
    "stats": (0, 0, 0),
    "index_sig": None,
    "uploader_key": 0,
    "skipped": [],
    "regen": False,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def clear_index():
    st.session_state.vectorstore = None
    st.session_state.index_owner = None
    st.session_state.files_meta = []
    st.session_state.stats = (0, 0, 0)
    st.session_state.index_sig = None
    st.session_state.skipped = []
    st.session_state.uploader_key += 1


def new_chat(title: str = "New chat") -> str:
    cid = uuid.uuid4().hex[:10]
    st.session_state.chats[cid] = {
        "id": cid,
        "title": title,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": "rag",
        "messages": [],
    }
    st.session_state.current = cid
    clear_index()
    return cid


def current_chat() -> dict:
    if not st.session_state.current or st.session_state.current not in st.session_state.chats:
        new_chat()
    return st.session_state.chats[st.session_state.current]


def index_for_current() -> bool:
    """True when the in-memory FAISS index belongs to the open chat."""
    return (
        st.session_state.vectorstore is not None
        and st.session_state.index_owner == st.session_state.current
    )


def safe_name(name: str) -> str:
    """Strip path parts and keep a short, printable filename."""
    base = Path(name).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    return (base[:120] or "file").strip()


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------- #
# API key gate
# --------------------------------------------------------------------------- #
def validate_openai_key(key: str) -> tuple[bool, str]:
    from openai import AuthenticationError, OpenAI
    try:
        OpenAI(api_key=key, timeout=20).models.list()
        return True, ""
    except AuthenticationError:
        return False, "That key was rejected by OpenAI. Check it and try again."
    except Exception as e:
        return False, f"Could not verify the key: {e}"


def api_key_gate():
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
        "[data-testid='collapsedControl'] { display: none; }</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{esc(BRAND)}</h1>
            <p>A practice project that combines a custom chatbot with document RAG.
            Answers from your files come with sources. General chat is available too.</p>
            <div class="pills">
                <span class="pill">PDF</span><span class="pill">DOCX</span>
                <span class="pill">TXT</span><span class="pill">JSON</span>
                <span class="pill">MD</span><span class="pill">CSV</span>
                <span class="pill">RAG</span><span class="pill">Chat</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2.4, 1])
    with mid:
        with st.form("api_form"):
            st.markdown("**Enter your OpenAI API key to continue**")
            key = st.text_input("API key", type="password", placeholder="sk-...", label_visibility="collapsed")
            go = st.form_submit_button("Continue", use_container_width=True, type="primary")
        if go:
            if not key.strip():
                st.error("Paste your API key first.")
            else:
                with st.spinner("Checking your key..."):
                    ok, msg = validate_openai_key(key.strip())
                if ok:
                    st.session_state.api_key = key.strip()
                    if not st.session_state.chats:
                        new_chat()
                    st.rerun()
                else:
                    st.error(msg)
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and st.button("Use key from .env", use_container_width=True):
            with st.spinner("Checking your key..."):
                ok, msg = validate_openai_key(env_key)
            if ok:
                st.session_state.api_key = env_key
                if not st.session_state.chats:
                    new_chat()
                st.rerun()
            else:
                st.error(msg)
        st.markdown(
            '<p class="gate-note">Your key stays in this browser session only. '
            "It is never written to disk or shared with other visitors.</p>",
            unsafe_allow_html=True,
        )
    st.stop()


if not st.session_state.api_key:
    api_key_gate()

api_key = st.session_state.api_key

# --------------------------------------------------------------------------- #
# Document loaders
# --------------------------------------------------------------------------- #
def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def flatten_json(obj, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines += flatten_json(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines += flatten_json(v, f"{prefix}[{i}]")
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def load_file(name: str, raw: bytes) -> list[Document]:
    name = safe_name(name)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    docs: list[Document] = []

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": name, "label": f"{name}, p. {i + 1}"},
                ))
    elif ext == "docx":
        from docx import Document as DocxDocument
        d = DocxDocument(io.BytesIO(raw))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        if parts:
            docs.append(Document(page_content="\n".join(parts), metadata={"source": name, "label": name}))
    elif ext in {"txt", "md", "csv"}:
        text = decode_text(raw).strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "label": name}))
    elif ext == "json":
        text = "\n".join(flatten_json(json.loads(decode_text(raw))))
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name, "label": name}))
    return docs


def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)


def build_vectorstore(files: list[tuple[str, bytes]], size: int, overlap: int):
    all_docs: list[Document] = []
    skipped: list[str] = []
    total_bytes = 0

    if len(files) > MAX_FILES:
        return None, 0, 0, [f"Too many files (max {MAX_FILES})."]

    for name, raw in files:
        name = safe_name(name)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ALLOWED_EXT:
            skipped.append(f"{name} (type not allowed)")
            continue
        if len(raw) > MAX_FILE_BYTES:
            skipped.append(f"{name} (over {MAX_FILE_BYTES // (1024 * 1024)} MB)")
            continue
        total_bytes += len(raw)
        try:
            loaded = load_file(name, raw)
        except Exception as e:
            skipped.append(f"{name} ({e.__class__.__name__})")
            continue
        if not loaded:
            skipped.append(f"{name} (no readable text)")
        all_docs += loaded

    if not all_docs:
        return None, 0, 0, skipped

    n_chars = sum(len(d.page_content) for d in all_docs)
    if n_chars > MAX_TOTAL_CHARS:
        return None, 0, n_chars, skipped + [f"Documents too large (max {MAX_TOTAL_CHARS:,} characters)."]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    if len(chunks) > MAX_CHUNKS:
        return None, len(chunks), n_chars, skipped + [f"Too many chunks ({len(chunks)}; max {MAX_CHUNKS}). Lower chunk size or remove files."]

    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks), n_chars, skipped


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f'<div class="side-title">{esc(BRAND)}</div>', unsafe_allow_html=True)

    if st.button("+  New chat", type="primary", use_container_width=True, key="new_chat"):
        new_chat()
        st.rerun()

    st.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
    _chat = current_chat()
    mode_options = ["Documents (RAG)", "General chat"]
    mode_key = f"mode_{_chat['id']}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = (
            mode_options[0] if _chat.get("mode", "rag") == "rag" else mode_options[1]
        )
    mode = st.radio(
        "Answer mode",
        mode_options,
        label_visibility="collapsed",
        help="RAG answers only from your uploaded files. General chat uses the model freely.",
        key=mode_key,
    )
    _chat["mode"] = "rag" if mode.startswith("Documents") else "chat"

    st.markdown('<div class="side-label">History</div>', unsafe_allow_html=True)
    ordered = sorted(st.session_state.chats.values(), key=lambda c: c["created"], reverse=True)
    for c in ordered:
        active = c["id"] == st.session_state.current
        title = c["title"] if len(c["title"]) <= 28 else c["title"][:27] + "..."
        a, b = st.columns([5, 1])
        btn_key = f"chat_on_{c['id']}" if active else f"chat_off_{c['id']}"
        if a.button(title, key=btn_key, use_container_width=True):
            if st.session_state.current != c["id"]:
                st.session_state.current = c["id"]
                # Do not leak another chat's index into this one
                if st.session_state.index_owner != c["id"]:
                    clear_index()
            st.rerun()
        if b.button("x", key=f"del_{c['id']}", help="Delete this chat"):
            if st.session_state.index_owner == c["id"]:
                clear_index()
            del st.session_state.chats[c["id"]]
            if not st.session_state.chats:
                new_chat()
            elif st.session_state.current == c["id"]:
                st.session_state.current = next(iter(st.session_state.chats))
                if st.session_state.index_owner != st.session_state.current:
                    clear_index()
            st.rerun()

    with st.expander("Settings", expanded=False):
        model_name = st.selectbox("Chat model", CHAT_MODELS)
        temperature = st.slider("Creativity (general chat)", 0.0, 1.2, 0.4, 0.1)
        persona = st.selectbox("Personality (general chat)", list(PERSONAS))
        custom_system = st.text_area(
            "Custom system role (optional)",
            placeholder="Optional instructions for general chat...",
            height=80,
        )

    with st.expander("Retrieval", expanded=False):
        chunk_size = st.slider("Chunk size", 400, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        # Keep overlap strictly below chunk size for the text splitter
        chunk_overlap = min(chunk_overlap, max(0, chunk_size - 50))
        top_k = st.slider("Chunks per answer (k)", 2, 10, 4)

    chat = current_chat()
    export_md = f"# {chat['title']}\n\n" + "\n\n".join(
        f"**{'You' if m['role'] == 'user' else BRAND}:** {m['content']}"
        for m in chat["messages"]
    )
    st.download_button(
        "Export chat (.md)",
        export_md,
        file_name="mycustomrag-chat.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not chat["messages"],
    )

    export_json = json.dumps(
        {"title": chat["title"], "mode": chat.get("mode", "rag"), "messages": [
            {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
            for m in chat["messages"]
        ]},
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "Export chat (.json)",
        export_json,
        file_name="mycustomrag-chat.json",
        mime="application/json",
        use_container_width=True,
        disabled=not chat["messages"],
    )

    imported = st.file_uploader("Import chat (.json)", type=["json"], key="import_chat")
    if imported is not None:
        try:
            data = json.loads(imported.getvalue().decode("utf-8"))
            msgs = data.get("messages", [])
            if not isinstance(msgs, list):
                raise ValueError("messages must be a list")
            clean = []
            for m in msgs[:200]:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content", "")
                if role in {"user", "assistant"} and isinstance(content, str):
                    entry = {"role": role, "content": content[:50_000]}
                    src = m.get("sources")
                    if isinstance(src, list):
                        entry["sources"] = [
                            {"label": str(s.get("label", ""))[:200], "text": str(s.get("text", ""))[:400]}
                            for s in src[:20] if isinstance(s, dict)
                        ]
                    clean.append(entry)
            cid = new_chat(str(data.get("title", "Imported chat"))[:60])
            st.session_state.chats[cid]["messages"] = clean
            st.session_state.chats[cid]["mode"] = "chat" if data.get("mode") == "chat" else "rag"
            st.success("Chat imported. Re-upload documents if you need RAG again.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not import that file: {e}")

    if st.button("Change API key", use_container_width=True):
        st.session_state.api_key = ""
        st.rerun()

    approx = sum(len(m.get("content", "")) for m in chat["messages"]) // 4
    st.caption(f"About {approx:,} tokens in this chat  ·  session only")

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
chat = current_chat()
is_rag = chat.get("mode", "rag") == "rag"

st.markdown(
    f"""
    <div class="hero">
        <h1>{esc(BRAND)}</h1>
        <p>Upload documents and ask questions with cited sources, or switch to general chat.
        History stays in your session. Export a chat if you want to keep it.</p>
        <div class="pills">
            <span class="pill">PDF</span><span class="pill">DOCX</span>
            <span class="pill">TXT</span><span class="pill">JSON</span>
            <span class="pill">MD</span><span class="pill">CSV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<span class="mode-chip">{esc(model_name)} · {"Documents (RAG)" if is_rag else "General chat"}</span>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Upload + index (RAG mode)
# --------------------------------------------------------------------------- #
uploaded = None
if is_rag:
    uploaded = st.file_uploader(
        "Upload your documents",
        type=sorted(ALLOWED_EXT),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"Up to {MAX_FILES} files, {MAX_FILE_BYTES // (1024 * 1024)} MB each.",
    )

if is_rag and uploaded:
    files_sig = tuple((safe_name(f.name), f.size) for f in uploaded)
    sig = (files_sig, chunk_size, chunk_overlap)
    if st.session_state.index_sig != sig:
        with st.spinner("Reading and indexing your files..."):
            try:
                store, n_chunks, n_chars, skipped = build_vectorstore(
                    [(f.name, f.getvalue()) for f in uploaded],
                    chunk_size,
                    chunk_overlap,
                )
            except Exception as e:
                msg = str(e)
                if "401" in msg or "invalid_api_key" in msg or "Incorrect API key" in msg:
                    st.error("OpenAI rejected your API key. Use Change API key in the sidebar.")
                else:
                    st.error(f"Indexing failed: {msg}")
                st.stop()

        st.session_state.index_sig = sig
        st.session_state.skipped = skipped
        if store is None:
            clear_index()
            st.error("Could not build an index from those files.")
            for s in skipped:
                st.warning(f"Skipped {s}")
        else:
            files_meta = [{"name": safe_name(f.name), "size": f.size} for f in uploaded]
            title = files_meta[0]["name"] + (f" +{len(files_meta) - 1}" if len(files_meta) > 1 else "")
            if chat["title"] in {"New chat", "Imported chat"} or not chat["messages"]:
                chat["title"] = title[:60]
            st.session_state.vectorstore = store
            st.session_state.index_owner = st.session_state.current
            st.session_state.files_meta = files_meta
            st.session_state.stats = (len(files_meta), n_chunks, n_chars)
            st.session_state.skipped = skipped
            # Clear old Q&A when re-indexing new files for this chat
            chat["messages"] = []
            st.rerun()

for s in st.session_state.skipped:
    st.warning(f"Skipped {s}")

if is_rag and index_for_current():
    n_files, n_chunks, n_chars = st.session_state.stats
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="stat"><div class="num">{n_files}</div><div class="lbl">Files indexed</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="stat rose"><div class="num">{n_chunks}</div><div class="lbl">Searchable chunks</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="stat gold"><div class="num">{n_chars:,}</div><div class="lbl">Characters read</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Your library", expanded=False):
        for f in st.session_state.files_meta:
            ext = f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
            label, color = TYPE_STYLE.get(ext, ("FILE", "#666666"))
            st.markdown(
                f'<div class="filerow"><span class="badge" style="background:{color}">{esc(label)}</span>'
                f'<span class="name">{esc(f["name"])}</span>'
                f'<span class="meta">{f["size"] / 1024:.1f} KB</span></div>',
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- #
# Prompts + generation
# --------------------------------------------------------------------------- #
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions strictly using the context taken from the user's documents.\n"
     "Rules:\n"
     "1. Use ONLY the context below. Do not use outside knowledge.\n"
     "2. If the answer is not in the context, say: \"I couldn't find that in the documents.\"\n"
     "3. Cite sources using the bracketed labels, e.g. (report.pdf, p. 3).\n"
     "4. If several files are relevant, combine them and cite each one.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('label', d.metadata.get('source', 'doc'))}]\n{d.page_content}" for d in docs)


def chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in chunk)
    return str(chunk)


def render_sources(sources: list[dict]):
    with st.expander(f"Sources ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(
                f'<div class="src"><div class="tag">{esc(s["label"])}</div>{esc(s["text"])}</div>',
                unsafe_allow_html=True,
            )


def general_system() -> str:
    base = (custom_system or "").strip() or PERSONAS[persona]
    return f"Your name is {BRAND}. {base}"


def generate_rag(question: str) -> tuple[str, list[dict]] | None:
    if not index_for_current():
        st.error("Upload and index documents first.")
        return None
    store = st.session_state.vectorstore
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key, streaming=True)
    chain = RAG_PROMPT | llm

    with st.chat_message("assistant"):
        slot = st.empty()
        slot.markdown(TYPING_HTML, unsafe_allow_html=True)
        answer = ""
        try:
            docs = retriever.invoke(question)
            for piece in chain.stream({"context": format_docs(docs), "question": question}):
                answer += chunk_text(piece.content)
                slot.markdown(answer + " ▌")
            slot.markdown(answer)
        except Exception as e:
            slot.empty()
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg:
                st.error("OpenAI rejected your API key. Use Change API key in the sidebar.")
            else:
                st.error(f"Something went wrong: {msg}")
            return None
        sources = [
            {
                "label": d.metadata.get("label", d.metadata.get("source", "doc")),
                "text": d.page_content[:400].replace("\n", " "),
            }
            for d in docs
        ]
        render_sources(sources)
    return answer, sources


def generate_chat(question: str, history: list[dict]) -> str | None:
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, streaming=True)
    msgs = [SystemMessage(content=general_system())]
    for m in history[-MAX_HISTORY_MSGS:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=question))

    with st.chat_message("assistant"):
        slot = st.empty()
        slot.markdown(TYPING_HTML, unsafe_allow_html=True)
        answer = ""
        try:
            for piece in llm.stream(msgs):
                if piece.content:
                    answer += chunk_text(piece.content)
                    slot.markdown(answer + " ▌")
            slot.markdown(answer)
        except Exception as e:
            slot.empty()
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg:
                st.error("OpenAI rejected your API key. Use Change API key in the sidebar.")
            else:
                st.error(f"Something went wrong: {msg}")
            return None
    return answer


# --------------------------------------------------------------------------- #
# Chat area
# --------------------------------------------------------------------------- #
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

ready_for_rag = is_rag and index_for_current()
ready_for_chat = not is_rag

if is_rag and not ready_for_rag and not uploaded:
    st.markdown(
        '<div class="empty"><b>Nothing indexed yet.</b><br>'
        "Upload PDF, DOCX, TXT, JSON, MD or CSV files above, then ask a question.</div>",
        unsafe_allow_html=True,
    )
elif not is_rag and not chat["messages"]:
    st.markdown(
        '<div class="empty"><b>General chat</b><br>'
        "Ask anything. Switch to Documents (RAG) in the sidebar when you want cited answers from files.</div>",
        unsafe_allow_html=True,
    )

prompt_label = "Ask something about your documents..." if is_rag else f"Message {BRAND}..."
question = None
if ready_for_rag or ready_for_chat:
    question = st.chat_input(prompt_label)

should_run = False
q_text = ""

if st.session_state.pop("regen", False) and chat["messages"]:
    # Regenerate last assistant reply from the previous user message
    if chat["messages"][-1]["role"] == "assistant":
        chat["messages"].pop()
    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        q_text = chat["messages"][-1]["content"]
        should_run = True
        # remove the user message so we re-append cleanly below
        chat["messages"].pop()

elif question and question.strip():
    q_text = question.strip()[:8_000]
    should_run = True

if should_run and q_text:
    if is_rag and not ready_for_rag:
        st.error("Index documents before asking.")
        st.stop()

    chat["messages"].append({"role": "user", "content": q_text})
    with st.chat_message("user"):
        st.markdown(q_text)

    if chat["title"] == "New chat":
        chat["title"] = q_text[:48] + ("..." if len(q_text) > 48 else "")

    if is_rag:
        result = generate_rag(q_text)
        if result is None:
            chat["messages"].pop()  # drop the user turn on failure so Retry works cleanly
        else:
            answer, sources = result
            chat["messages"].append({"role": "assistant", "content": answer, "sources": sources})
    else:
        # history without the just-added user message (generate_chat appends it)
        prior = chat["messages"][:-1]
        answer = generate_chat(q_text, prior)
        if answer is None:
            chat["messages"].pop()
        else:
            chat["messages"].append({"role": "assistant", "content": answer})

if chat["messages"]:
    last = chat["messages"][-1]
    label = "Retry" if last["role"] == "user" else "Regenerate"
    if st.button(label, key="regen_btn"):
        st.session_state.regen = True
        st.rerun()
