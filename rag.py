import sqlite3, json, textwrap, time, re, sys, threading
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Config ── 
OLLAMA_URL      = "http://localhost:11434"
EMBED_MODEL     = "all-minilm"
DB_PATH         = "rag.db"