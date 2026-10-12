import re
from typing import List
from pydantic import BaseModel
from src.epub_parser import BookDocument, Chapter

PARAGRAPH_SEPARATOR = "\n\n"

class BookChunk(BaseModel):
    chunk_id: str
    book_title: str
    chapter_index: int
    chapter_title: str
    chunk_index: int
    text: str
    embed_text: str
    word_count: int


def __split_into_sentences(text: str) -> List[str]:
    """Splits a paragraph into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def __chunk_chapter(chapter: Chapter, book_title: str, chunk_size: int, overlap: int) -> List[BookChunk]:
    """
    Splits a single chapter into overlapping narrative chunks.
    Preserves paragraph and sentence boundaries.
    """
    paragraphs = [p.strip() for p in chapter.content.split(PARAGRAPH_SEPARATOR) if p.strip()]
    if not paragraphs:
        return []

    # Flatten into sentence units while keeping paragraph boundary markers
    units: List[str] = []
    for p in paragraphs:
        sentences = __split_into_sentences(p)
        if sentences:
            units.extend(sentences)
            units.append(PARAGRAPH_SEPARATOR)

    chunks: List[BookChunk] = []
    current_words: List[str] = []
    current_text_parts: List[str] = []
    chunk_idx = 1

    i = 0
    while i < len(units):
        unit = units[i]

        if unit == PARAGRAPH_SEPARATOR:
            current_text_parts.append(unit)
            i += 1
            continue

        unit_words = unit.split()
        current_words.extend(unit_words)
        current_text_parts.append(unit + " ")

        # Check if we have accumulated enough words for a chunk
        if len(current_words) >= chunk_size:
            raw_text = "".join(current_text_parts).strip()
            # Contextual embedding
            embed_text = f"[Book: {book_title} | Chapter: {chapter.title}]\n\n{raw_text}"

            clean_book_id = re.sub(r"[^a-zA-Z0-9_-]", "_", book_title)[:30]
            chunk_id = f"{clean_book_id}_c{chapter.index}_p{chunk_idx}"

            chunks.append(
                BookChunk(
                    chunk_id=chunk_id,
                    book_title=book_title,
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    chunk_index=chunk_idx,
                    text=raw_text,
                    embed_text=embed_text,
                    word_count=len(current_words),
                )
            )
            chunk_idx += 1

            # Backtrack to maintain overlap
            overlap_words_target = overlap
            backtrack_words = 0
            backtrack_idx = i

            while backtrack_idx > 0 and backtrack_words < overlap_words_target:
                prev_unit = units[backtrack_idx]
                if prev_unit != PARAGRAPH_SEPARATOR:
                    backtrack_words += len(prev_unit.split())
                backtrack_idx -= 1

            # Reset current accumulator with overlap content
            current_words = []
            current_text_parts = []
            i = max(backtrack_idx + 1, backtrack_idx)
            continue

        i += 1

    # Remaining tail chunk
    if current_words:
        raw_text = "".join(current_text_parts).strip()
        if len(raw_text.split()) > 25:  # Avoid tiny trailing fragments
            embed_text = f"[Book: {book_title} | Chapter: {chapter.title}]\n\n{raw_text}"
            clean_book_id = re.sub(r"[^a-zA-Z0-9_-]", "_", book_title)[:30]
            chunk_id = f"{clean_book_id}_c{chapter.index}_p{chunk_idx}"

            chunks.append(
                BookChunk(
                    chunk_id=chunk_id,
                    book_title=book_title,
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    chunk_index=chunk_idx,
                    text=raw_text,
                    embed_text=embed_text,
                    word_count=len(current_words),
                )
            )

    return chunks


def chunk_book(book: BookDocument, chunk_size: int, overlap: int) -> List[BookChunk]:
    """Chunks an entire Book chapter by chapter."""
    all_chunks: List[BookChunk] = []
    for chapter in book.chapters:
        chapter_chunks = __chunk_chapter(
            chapter=chapter,
            book_title=book.title,
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(chapter_chunks)
    return all_chunks
