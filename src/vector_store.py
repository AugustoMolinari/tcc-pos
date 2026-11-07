import chromadb
import torch
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.chunker import BookChunk


class BookVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        embedding_model_name: str,
        reranker_model_name: str
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self._embedder = None
        self._reranker = None

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="books_rag",
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        return self._reranker

    def _get_query_prefix(self) -> str:
        """Instruction prefix required for BGE family models during search."""
        if "bge" in self.embedding_model_name.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""

    def add_book_chunks(self, chunks: List[BookChunk], batch_size: int = 64) -> int:
        """Embeds and indexes book chunks in batches."""
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [c.chunk_id for c in batch]
            texts_to_embed = [c.embed_text for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "book_title": c.book_title,
                    "chapter_index": c.chapter_index,
                    "chapter_title": c.chapter_title,
                    "chunk_index": c.chunk_index,
                    "word_count": c.word_count,
                }
                for c in batch
            ]

            embeddings = self.embedder.encode(
                texts_to_embed,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            total_added += len(batch)

        return total_added

    def list_indexed_books(self) -> List[Dict[str, Any]]:
        """Returns metadata for all unique books currently indexed."""

        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        books: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            if not m:
                continue
            title = m.get("book_title", "Unknown")
            if title not in books:
                books[title] = {
                    "book_title": title,
                    "chapters": set(),
                    "total_chunks": 0
                }
            books[title]["chapters"].add(m.get("chapter_index", 1))
            books[title]["total_chunks"] += 1

        formatted = []
        for title, info in books.items():
            formatted.append({
                "book_title": title,
                "chapter_count": len(info["chapters"]),
                "total_chunks": info["total_chunks"]
            })

        return sorted(formatted, key=lambda x: x["book_title"])

    def delete_book(self, book_title: str) -> int:
        """Deletes all chunks for a given book title."""

        results = self.collection.get(where={"book_title": book_title})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def search(
        self,
        query: str,
        top_k: int,
        fetch_k: int,
        book_title: str
    ) -> List[Dict[str, Any]]:
        """
        Two-stage search:
        1. Fast vector similarity search over-retrieves candidate chunks (fetch_k).
        2. Cross-encoder reranks the candidates and returns the absolute best top_k.
        """
        fetch_k = max(fetch_k, top_k)

        prefixed_query = f"{self._get_query_prefix()}{query}"

        query_embedding = self.embedder.encode(
            [prefixed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        where_filter = {"book_title": book_title} if book_title else None

        # Step 1: Over-retrieve candidates from ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas"]
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Step 2: Construct query-document pairs for the Cross-Encoder
        pairs = [[query, doc] for doc in docs]

        # Predict relevance scores (logits)
        raw_scores = self.reranker.predict(pairs, convert_to_numpy=True)

        # Step 3: Bundle results and sort by reranker score descending
        scored_results = []
        for doc, meta, score in zip(docs, metas, raw_scores):
            scored_results.append({
                "text": doc,
                "book_title": meta.get("book_title", ""),
                "chapter_index": meta.get("chapter_index", 0),
                "chapter_title": meta.get("chapter_title", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score": float(score),
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]