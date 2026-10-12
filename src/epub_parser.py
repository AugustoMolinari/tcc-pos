import re
import warnings
import ebooklib
from pathlib import Path
from typing import List
from pydantic import BaseModel
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

# Suppress ebooklib and BeautifulSoup XML parsing warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib.epub")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib.epub")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class Chapter(BaseModel):
    index: int
    title: str
    content: str
    word_count: int


class BookDocument(BaseModel):
    title: str
    file_path: str
    chapters: List[Chapter]
    total_words: int


def __clean_html_to_text(html_bytes: bytes) -> str:
    """Converts HTML/XHTML bytes into clean, readable text preserving paragraphs."""
    soup = BeautifulSoup(html_bytes, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.extract()

    # Replace line breaks and paragraph tags with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    for p in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"]):
        p.append("\n\n")

    text = soup.get_text()

    # Normalize multiple whitespaces and excessive newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def __extract_chapter_title(soup: BeautifulSoup, fallback_title: str) -> str:
    """Attempts to find a clean chapter heading from HTML elements."""
    heading = soup.find(["h1", "h2", "h3"])
    if heading:
        h_text = heading.get_text().strip()
        if h_text and len(h_text) < 120:
            return h_text
    return fallback_title


from urllib.parse import unquote


def __build_toc_map(book: epub.EpubBook) -> dict:
    """Maps document file paths (href) to canonical titles from the Table of Contents."""
    toc_map = {}

    def _process_item(item):
        if isinstance(item, epub.Link):
            href = item.href.split("#")[0]
            clean_href = unquote(href)
            if clean_href and clean_href not in toc_map:
                toc_map[clean_href] = item.title
            fname = Path(clean_href).name
            if fname not in toc_map:
                toc_map[fname] = item.title
        elif isinstance(item, tuple) and len(item) == 2:
            if hasattr(item[0], "href"):
                href = item[0].href.split("#")[0]
                clean_href = unquote(href)
                if clean_href and clean_href not in toc_map:
                    toc_map[clean_href] = item[0].title
                fname = Path(clean_href).name
                if fname not in toc_map:
                    toc_map[fname] = item[0].title
            for sub in item[1]:
                _process_item(sub)

    for item in book.toc:
        _process_item(item)

    return toc_map


def parse_epub(epub_path: str | Path) -> BookDocument:
    """
    Parses an EPUB file, extracts metadata and structured chapters.
    Filters out empty pages and image-only pages.
    """
    path = Path(epub_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"EPUB file not found: {path}")

    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    toc_map = __build_toc_map(book)

    # Extract metadata
    titles = book.get_metadata("DC", "title")
    title = titles[0][0] if titles else path.stem

    # Build spine ordering map
    spine_ids = [item[0] for item in book.spine]

    # Process document items in spine order
    chapters: List[Chapter] = []
    chapter_index = 1
    total_words = 0

    # Retrieve all document items
    doc_items = {item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}

    # Iterate through spine
    for item_id in spine_ids:
        item = doc_items.get(item_id)
        if not item:
            continue

        raw_content = item.get_content()
        soup = BeautifulSoup(raw_content, "lxml")
        clean_text = __clean_html_to_text(raw_content)

        # Ignore tiny boilerplate pages (e.g. copyright, title-only, under 50 words)
        words = clean_text.split()
        if len(words) < 50:
            continue

        item_name = item.get_name()
        item_fname = Path(item_name).name

        # Priority 1: Title from the book's official Table of Contents
        if item_name in toc_map:
            title_candidate = toc_map[item_name]
        elif item_fname in toc_map:
            title_candidate = toc_map[item_fname]
        else:
            # Priority 2: Extract heading from HTML or fallback
            title_candidate = __extract_chapter_title(soup, f"Section {chapter_index}")

        # Clean title candidate (remove common whitespace artifacts)
        title_candidate = re.sub(r"\s+", " ", title_candidate).strip()
        if not title_candidate:
            title_candidate = f"Section {chapter_index}"

        chapter = Chapter(
            index=chapter_index,
            title=title_candidate,
            content=clean_text,
            word_count=len(words)
        )
        chapters.append(chapter)
        total_words += len(words)
        chapter_index += 1

    # If no chapters were found via spine, fallback to all document items
    if not chapters:
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            raw_content = item.get_content()
            clean_text = __clean_html_to_text(raw_content)
            words = clean_text.split()
            if len(words) < 50:
                continue

            soup = BeautifulSoup(raw_content, "lxml")
            title_candidate = __extract_chapter_title(soup, f"Chapter {chapter_index}")
            chapter = Chapter(
                index=chapter_index,
                title=title_candidate,
                content=clean_text,
                word_count=len(words)
            )
            chapters.append(chapter)
            total_words += len(words)
            chapter_index += 1

    return BookDocument(
        title=title,
        file_path=str(path),
        chapters=chapters,
        total_words=total_words
    )
