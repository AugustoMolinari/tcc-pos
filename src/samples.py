import httpx
from pathlib import Path
from typing import Dict

PUBLIC_DOMAIN_BOOKS: Dict[str, Dict[str, str]] = {
    "alice": {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "filename": "alices_adventures_in_wonderland.epub",
        "url": "https://www.gutenberg.org/ebooks/11.epub3.images",
        "fallback_url": "https://www.gutenberg.org/ebooks/11.epub.noimages"
    },
    "sherlock": {
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "filename": "the_adventures_of_sherlock_holmes.epub",
        "url": "https://www.gutenberg.org/ebooks/1661.epub3.images",
        "fallback_url": "https://www.gutenberg.org/ebooks/1661.epub.noimages"
    },
    "frankenstein": {
        "title": "Frankenstein; Or, The Modern Prometheus",
        "author": "Mary Wollstonecraft Shelley",
        "filename": "frankenstein.epub",
        "url": "https://www.gutenberg.org/ebooks/84.epub3.images",
        "fallback_url": "https://www.gutenberg.org/ebooks/84.epub.noimages"
    }
}


def download_sample_book(key: str, dest_dir: Path) -> Path:
    """Downloads a public domain book EPUB to the destination directory."""
    if key not in PUBLIC_DOMAIN_BOOKS:
        raise ValueError(f"Unknown book key: {key}. Available: {list(PUBLIC_DOMAIN_BOOKS.keys())}")

    book_info = PUBLIC_DOMAIN_BOOKS[key]
    dest_path = dest_dir / book_info["filename"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return dest_path

    headers = {"User-Agent": "BookRAG/1.0 (Public Domain Educational Assistant)"}

    for url in [book_info["url"], book_info.get("fallback_url")]:
        if not url:
            continue
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0, headers=headers) as client:
                resp = client.get(url)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    dest_path.write_bytes(resp.content)
                    return dest_path
        except Exception:
            continue

    raise RuntimeError(f"Failed to download book '{book_info['title']}' from Project Gutenberg.")
