import os
from pathlib import Path
from pydantic import BaseModel

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
SAMPLES_DIR = DATA_DIR / "samples"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
DEFAULT_BOOKS_DIR = Path(os.path.expanduser("~/books"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

# Default LLM Settings
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "llama3.2:3b"
TEMPERATURE = 0.3

# RAG & Chunking
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
VECTOR_FETCH_K = 20

class AppConfig(BaseModel):
    ollama_url: str = DEFAULT_OLLAMA_URL
    active_model: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    temperature: float = TEMPERATURE
    books_dir: Path = DEFAULT_BOOKS_DIR
    chroma_dir: Path = CHROMA_DIR
    samples_dir: Path = SAMPLES_DIR
    benchmarks_dir: Path = BENCHMARKS_DIR
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K_RESULTS
    fetch_k: int = VECTOR_FETCH_K

config = AppConfig()
