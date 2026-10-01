import os
from pathlib import Path
from pydantic import BaseModel

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
SAMPLES_DIR = DATA_DIR / "samples"
DEFAULT_BOOKS_DIR = Path(os.path.expanduser("~/books"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Default LLM Settings
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "llama3.2:3b"
TEMPERATURE = 0.3

# Default Embedding Model
# Fast, lightweight English: "BAAI/bge-small-en-v1.5" or "all-MiniLM-L6-v2"
# Multilingual alternative: "BAAI/bge-m3" or "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RAG & Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 7

class AppConfig(BaseModel):
    ollama_url: str = DEFAULT_OLLAMA_URL
    active_model: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    temperature: float = TEMPERATURE
    books_dir: Path = DEFAULT_BOOKS_DIR
    chroma_dir: Path = CHROMA_DIR
    samples_dir: Path = SAMPLES_DIR
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K_RESULTS

config = AppConfig()
