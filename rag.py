import sqlite3, json, textwrap, time, re, sys, threading
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup

#Config 
OLLAMA_URL      = "http://localhost:11434"
EMBED_MODEL     = "all-minilm"
DB_PATH         = "rag.db"
CHUNK_SIZE      = 120
CHUNK_OVERLAP   = 15
MAX_CHUNK_CHARS = 700
TOP_K           = 4
MAX_CTX_CHARS   = 2000
MIN_TEXT_CHARS  = 200

STATE = {"chat_model": "smollm2:135m", "timeout": 180}

#Supported file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".log", ".sql",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".docx", ".pdf", ".csv", ".xlsx", ".xls", ".pptx",
    ".html", ".htm", ".json", ".xml", ".epub", ".rtf",
}

#Ollama

def ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

def embed(text: str) -> list[float]:
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]

def _live_clock(label: str, stop_event: threading.Event):
    """Print a live elapsed timer on a single line until stop_event is set."""
    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        sys.stdout.write(f"\r  {label} [{mins:02d}:{secs:02d}] ...")
        sys.stdout.flush()
        time.sleep(0.5)
    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    sys.stdout.write(f"\r  {label} — done in {mins:02d}:{secs:02d}        \n")
    sys.stdout.flush()


def generate(prompt: str) -> str:
    stop = threading.Event()
    clock = threading.Thread(
        target=_live_clock,
        args=(f"Generating [{STATE['chat_model']}]", stop),
        daemon=True,
    )
    clock.start()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": STATE["chat_model"], "prompt": prompt, "stream": False},
            timeout=STATE["timeout"],
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    finally:
        stop.set()
        clock.join()



#Database

def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            source   TEXT,
            content  TEXT,
            vector   BLOB,
            added_at TEXT DEFAULT (datetime('now'))
        )
    """)
    con.commit()
    return con

def save_chunk(con, source: str, content: str, vector: list[float]):
    con.execute("INSERT INTO chunks (source, content, vector) VALUES (?,?,?)",
                (source, content, json.dumps(vector).encode()))
    con.commit()

def load_all(con) -> list[dict]:
    rows = con.execute("SELECT id, source, content, vector FROM chunks").fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2],
             "vector": np.array(json.loads(r[3]), dtype=np.float32)} for r in rows]

def delete_chunk(con, chunk_id: int):
    con.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
    con.commit()

def delete_source(con, source: str) -> int:
    cur = con.execute("DELETE FROM chunks WHERE source=?", (source,))
    con.commit()
    return cur.rowcount

def list_chunks(con):
    return con.execute(
        "SELECT id, source, substr(content,1,80), added_at FROM chunks ORDER BY id"
    ).fetchall()

def sources(con):
    return con.execute(
        "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
    ).fetchall()

def chunk_count(con) -> int:
    return con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]









