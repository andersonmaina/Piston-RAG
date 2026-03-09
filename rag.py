import sqlite3, json, textwrap, time, re, sys, threading
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Config ── 
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