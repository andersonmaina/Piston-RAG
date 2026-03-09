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

# Supported file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".log", ".sql",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".docx", ".pdf", ".csv", ".xlsx", ".xls", ".pptx",
    ".html", ".htm", ".json", ".xml", ".epub", ".rtf",
}


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