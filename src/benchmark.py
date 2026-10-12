import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from src.config import config
from src.epub_parser import parse_epub
from src.chunker import chunk_book
from src.vector_store import BookVectorStore
from src.llm import OllamaClient
from src.samples import get_alice_book_info, download_alice_book


# ------------------------------------------------------------------------------
# Golden Benchmark Dataset for Alice's Adventures in Wonderland
# ------------------------------------------------------------------------------

ALICE_GOLDEN_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "alice_q1",
        "chapter_target": "Chapter I",
        "chapter_name": "Down the Rabbit-Hole",
        "question": "What happens after Alice drinks from the little bottle on the glass table with the paper label 'DRINK ME'?",
        "expected_answer": "Alice shrinks down to ten inches high so that she can fit through the little door into the garden.",
        "keywords": ["shrink", "ten inches", "bottle", "door", "drink me"]
    },
    {
        "id": "alice_q2",
        "chapter_target": "Chapter II",
        "chapter_name": "The Pool of Tears",
        "question": "What causes the large pool of salt water that Alice slips into in Chapter 2, and what small animal does she swim near?",
        "expected_answer": "The pool was formed by Alice's own tears wept when she had grown to nine feet high. She swims near a mouse.",
        "keywords": ["tears", "wept", "nine feet", "mouse", "swim"]
    },
    {
        "id": "alice_q3",
        "chapter_target": "Chapter III",
        "chapter_name": "A Caucus-Race and a Long Tale",
        "question": "What kind of race does the Dodo propose to get the wet creatures dry, and who wins the race?",
        "expected_answer": "The Dodo proposes a Caucus-race where everyone runs and stops when they like, and the Dodo declares that everybody has won and all must have prizes.",
        "keywords": ["caucus-race", "dodo", "prizes", "dry", "won"]
    },
    {
        "id": "alice_q4",
        "chapter_target": "Chapter IV",
        "chapter_name": "The Rabbit Sends in a Little Bill",
        "question": "Why does the White Rabbit send Bill the Lizard down the chimney, and what happens when he comes down?",
        "expected_answer": "The Rabbit sends Bill down the chimney because Alice has grown huge inside the room, and Alice kicks Bill back up the chimney with her foot.",
        "keywords": ["bill", "lizard", "chimney", "kick", "grown"]
    },
    {
        "id": "alice_q5",
        "chapter_target": "Chapter V",
        "chapter_name": "Advice from a Caterpillar",
        "question": "What does the Caterpillar tell Alice about the mushroom and how it changes her size?",
        "expected_answer": "The Caterpillar tells Alice that one side of the mushroom will make her grow taller, and the other side will make her grow shorter.",
        "keywords": ["mushroom", "one side", "taller", "shorter", "caterpillar"]
    },
    {
        "id": "alice_q6",
        "chapter_target": "Chapter VI",
        "chapter_name": "Pig and Pepper",
        "question": "What unusual creature does Alice encounter at the Duchess's house that slowly vanishes until only its grin remains?",
        "expected_answer": "The Cheshire Cat, which disappears gradually beginning with the end of the tail and ending with its grin.",
        "keywords": ["cheshire cat", "cat", "grin", "vanish"]
    },
    {
        "id": "alice_q7",
        "chapter_target": "Chapter VII",
        "chapter_name": "A Mad Tea-Party",
        "question": "Who are the characters sitting together at the table having tea under a tree in the Mad Tea-Party scene, and what famous riddle is asked?",
        "expected_answer": "The March Hare, the Hatter, and a Dormouse are sitting together. The Hatter asks 'Why is a raven like a writing-desk?'",
        "keywords": ["march hare", "hatter", "dormouse", "tea", "raven", "writing-desk"]
    },
    {
        "id": "alice_q8",
        "chapter_target": "Chapter VIII",
        "chapter_name": "The Queen's Croquet-Ground",
        "question": "What living creatures and objects are used as balls, mallets, and arches in the Queen of Hearts' croquet game?",
        "expected_answer": "Live hedgehogs are used as balls, live flamingos as mallets, and soldiers bend over to form the arches.",
        "keywords": ["hedgehogs", "flamingos", "soldiers", "croquet", "queen"]
    },
    {
        "id": "alice_q9",
        "chapter_target": "Chapter IX",
        "chapter_name": "The Mock Turtle's Story",
        "question": "What strange branches of arithmetic does the Mock Turtle claim to have studied at the bottom of the sea?",
        "expected_answer": "The Mock Turtle claims to have studied Ambition, Distraction, Uglification, and Derision.",
        "keywords": ["mock turtle", "arithmetic", "uglification", "derision", "distraction", "ambition"]
    },
    {
        "id": "alice_q10",
        "chapter_target": "Chapter XI",
        "chapter_name": "Who Stole the Tarts?",
        "question": "Who is accused and put on trial for stealing the Queen of Hearts' tarts?",
        "expected_answer": "The Knave of Hearts is on trial before the King and Queen for stealing the tarts.",
        "keywords": ["knave of hearts", "knave", "tarts", "trial", "queen"]
    }
]


# ------------------------------------------------------------------------------
# System & Hardware Metrics Helpers
# ------------------------------------------------------------------------------

def __get_gpu_memory_mb() -> Optional[float]:
    """Reads real-time GPU memory from nvidia-smi if available."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            if lines and "," in lines[0]:
                parts = lines[0].split(",")
                return float(parts[0].strip())
    except Exception:
        pass
    return None


def __get_ollama_model_vram_mb(llm: OllamaClient, model_name: str) -> Optional[float]:
    """Queries Ollama /api/ps to retrieve exact VRAM allocated to this model."""
    try:
        models = llm.get_loaded_models_info()
        for m in models:
            if m.get("name") == model_name or model_name.startswith(m.get("name", "")):
                size_vram = m.get("size_vram", 0)
                if size_vram > 0:
                    return round(size_vram / (1024 * 1024), 1)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------
# Quality Metrics Helpers
# ------------------------------------------------------------------------------

def __check_chapter_accuracy(target_chapter: str, chapter_name: str, generated_text: str) -> bool:
    """Checks whether the generated response cites the target chapter identifier or name."""
    text_lower = generated_text.lower()
    target_lower = target_chapter.lower()
    name_lower = chapter_name.lower()

    if target_lower in text_lower or name_lower in text_lower:
        return True

    roman_map = {
        "chapter i": "chapter 1",
        "chapter ii": "chapter 2",
        "chapter iii": "chapter 3",
        "chapter iv": "chapter 4",
        "chapter v": "chapter 5",
        "chapter vi": "chapter 6",
        "chapter vii": "chapter 7",
        "chapter viii": "chapter 8",
        "chapter ix": "chapter 9",
        "chapter x": "chapter 10",
        "chapter xi": "chapter 11",
        "chapter xii": "chapter 12"
    }
    alt_target = roman_map.get(target_lower)
    if alt_target and alt_target in text_lower:
        return True

    return False


def __calculate_keyword_coverage(keywords: List[str], generated_text: str) -> float:
    """Calculates what percentage of ground truth key concepts appear in the response."""
    if not keywords:
        return 1.0
    text_lower = generated_text.lower()
    matched = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(matched / len(keywords), 2)


def __calculate_semantic_similarity(embedder: SentenceTransformer, text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts using the embedding model."""
    try:
        e1 = embedder.encode(text1, normalize_embeddings=True)
        e2 = embedder.encode(text2, normalize_embeddings=True)
        sim = float(np.dot(e1, e2))
        return round(max(0.0, min(1.0, sim)), 3)
    except Exception:
        return 0.0


# ------------------------------------------------------------------------------
# Alice Book Ingestion Helper for Benchmark
# ------------------------------------------------------------------------------

def __ensure_benchmark_book_indexed(vs: BookVectorStore, console: Console) -> str:
    """
    Ensures that Alice's Adventures in Wonderland is downloaded and indexed in ChromaDB.
    Returns the resolved book title.
    """
    book_info = get_alice_book_info()
    target_title = book_info["title"]

    indexed = vs.list_indexed_books()
    for b in indexed:
        if b["book_title"].lower() == target_title.lower() or target_title.lower() in b["book_title"].lower():
            return b["book_title"]

    console.print(f"[cyan]Alice's Adventures in Wonderland is not yet indexed. Preparing now...[/cyan]")

    epub_path = config.samples_dir / book_info["filename"]
    if not epub_path.exists() or epub_path.stat().st_size < 1000:
        with console.status(f"[bold green]Downloading '{target_title}' from Project Gutenberg..."):
            epub_path = download_alice_book(config.samples_dir)
        console.print(f"[bold green]✓ Downloaded to:[/bold green] {epub_path}")

    with console.status(f"[bold green]Parsing chapters and indexing '{target_title}'..."):
        book_doc = parse_epub(epub_path)
        chunks = chunk_book(book_doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        vs.add_book_chunks(chunks)

    console.print(f"[bold green]✓ Indexed '{book_doc.title}' ({len(chunks)} chunks) ready for benchmarks![/bold green]\n")
    return book_doc.title


# ------------------------------------------------------------------------------
# Benchmark Execution Engine
# ------------------------------------------------------------------------------

def run_benchmark_on_model(
    model_name: str,
    questions: List[Dict[str, Any]],
    console: Optional[Console] = None
) -> Dict[str, Any]:
    """
    Executes benchmark questions against active model, records detailed latency,
    throughput, VRAM, and quality metrics.
    """
    console = console or Console()
    vs = BookVectorStore(persist_dir=config.chroma_dir, embedding_model_name=config.embedding_model)
    llm = OllamaClient(base_url=config.ollama_url, default_model=model_name, temperature=config.temperature)

    book_title = __ensure_benchmark_book_indexed(vs, console)

    question_results = []

    peak_vram_mb = 0.0

    console.print(f"\n[bold cyan]Starting Model Evaluation:[/bold cyan] [bold white]{model_name}[/bold white]")
    console.print(f"[dim]Testing against: {book_title} ({len(questions)} test items)[/dim]\n")

    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        q_text = item["question"]
        expected_ans = item["expected_answer"]
        target_ch = item["chapter_target"]
        ch_name = item["chapter_name"]
        keywords = item.get("keywords", [])

        console.print(f"[bold white][{idx}/{len(questions)}][/bold white] [cyan]{q_text}[/cyan]")

        # 1. Retrieval
        retrieval_start = time.perf_counter()
        retrieved_chunks = vs.search(query=q_text, top_k=config.top_k, book_title=book_title)
        retrieval_time_s = time.perf_counter() - retrieval_start

        # Monitor initial VRAM
        gpu_mem_before = __get_gpu_memory_mb()
        if gpu_mem_before and gpu_mem_before > peak_vram_mb:
            peak_vram_mb = gpu_mem_before

        # 2. LLM Inference via non-streaming chat with full metrics
        llm_start = time.perf_counter()
        try:
            resp = llm.chat_with_metrics(
                query=q_text,
                book_title=book_title,
                retrieved_chunks=retrieved_chunks,
                model=model_name
            )
            llm_elapsed_time_s = time.perf_counter() - llm_start
        except Exception as e:
            console.print(f"  [bold red]Error querying model:[/bold red] {e}")
            continue

        # Monitor peak VRAM during/after generation
        gpu_mem_after = __get_gpu_memory_mb()
        if gpu_mem_after and gpu_mem_after > peak_vram_mb:
            peak_vram_mb = gpu_mem_after

        # Also inspect Ollama allocated VRAM
        ollama_vram = __get_ollama_model_vram_mb(llm, model_name)
        if ollama_vram and ollama_vram > peak_vram_mb:
            peak_vram_mb = ollama_vram

        # Extract Ollama native execution statistics
        generated_answer = resp.get("message", {}).get("content", "").strip()
        prompt_eval_count = resp.get("prompt_eval_count", 0)
        prompt_eval_dur_ns = resp.get("prompt_eval_duration", 0)
        eval_count = resp.get("eval_count", 0)
        eval_dur_ns = resp.get("eval_duration", 0)
        total_dur_ns = resp.get("total_duration", 0)

        # Computations
        ttft_ms = round((prompt_eval_dur_ns / 1e6), 1) if prompt_eval_dur_ns > 0 else 0.0
        tokens_per_sec = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns > 0 else 0.0
        prompt_tok_per_sec = round(prompt_eval_count / (prompt_eval_dur_ns / 1e9), 1) if prompt_eval_dur_ns > 0 else 0.0
        total_latency_s = round((total_dur_ns / 1e9) + retrieval_time_s, 2) if total_dur_ns > 0 else round(llm_elapsed_time_s + retrieval_time_s, 2)

        # Quality scoring
        cited_chapter = __check_chapter_accuracy(target_ch, ch_name, generated_answer)
        kw_cov = __calculate_keyword_coverage(keywords, generated_answer)
        sem_sim = __calculate_semantic_similarity(vs.embedder, expected_ans, generated_answer)

        # Display concise line
        ch_status = "[green]✓ Cited[/green]" if cited_chapter else "[yellow]✗ Missing[/yellow]"
        console.print(
            f"   ↳ TTFT: [bold green]{ttft_ms} ms[/bold green] | "
            f"Speed: [bold green]{tokens_per_sec} tok/s[/bold green] | "
            f"Latency: [cyan]{total_latency_s}s[/cyan] | "
            f"Chapter: {ch_status} | "
            f"Sim: [magenta]{sem_sim * 100:.0f}%[/magenta]"
        )

        retrieved_chapter_names = list(set(c.get("chapter_title", "") for c in retrieved_chunks))

        question_results.append({
            "id": q_id,
            "question": q_text,
            "target_chapter": target_ch,
            "chapter_name": ch_name,
            "expected_answer": expected_ans,
            "generated_answer": generated_answer,
            "retrieved_chapters": retrieved_chapter_names,
            "metrics": {
                "ttft_ms": ttft_ms,
                "tokens_per_sec": tokens_per_sec,
                "prompt_tokens_per_sec": prompt_tok_per_sec,
                "prompt_tokens": prompt_eval_count,
                "eval_tokens": eval_count,
                "retrieval_time_s": round(retrieval_time_s, 3),
                "total_latency_s": total_latency_s,
                "cited_chapter": cited_chapter,
                "keyword_coverage": kw_cov,
                "semantic_similarity": sem_sim
            }
        })

    if not question_results:
        console.print("[bold red]Benchmark completed with no valid question responses.[/bold red]")
        return {}

    avg_ttft = round(sum(q["metrics"]["ttft_ms"] for q in question_results) / len(question_results), 1)
    avg_tps = round(sum(q["metrics"]["tokens_per_sec"] for q in question_results) / len(question_results), 1)
    avg_latency = round(sum(q["metrics"]["total_latency_s"] for q in question_results) / len(question_results), 2)
    chapter_acc = round(sum(1 for q in question_results if q["metrics"]["cited_chapter"]) / len(question_results) * 100, 1)
    avg_kw = round(sum(q["metrics"]["keyword_coverage"] for q in question_results) / len(question_results) * 100, 1)
    avg_sim = round(sum(q["metrics"]["semantic_similarity"] for q in question_results) / len(question_results), 3)

    timestamp_str = datetime.now().isoformat()
    clean_model_name = model_name.replace(":", "_").replace("/", "_")

    summary_data = {
        "model": model_name,
        "book": book_title,
        "timestamp": timestamp_str,
        "total_questions": len(question_results),
        "avg_ttft_ms": avg_ttft,
        "avg_tokens_per_sec": avg_tps,
        "avg_latency_s": avg_latency,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "chapter_accuracy_pct": chapter_acc,
        "avg_keyword_coverage_pct": avg_kw,
        "avg_semantic_similarity": avg_sim,
    }

    full_run_record = {
        "summary": summary_data,
        "questions": question_results
    }

    __save_benchmark_results(clean_model_name, full_run_record)

    return full_run_record


# ------------------------------------------------------------------------------
# JSON Persistence & Model Comparison Index
# ------------------------------------------------------------------------------

def __save_benchmark_results(clean_model_name: str, full_record: Dict[str, Any]):
    """Saves per-run detail JSON and updates comparison.json."""
    config.benchmarks_dir.mkdir(parents=True, exist_ok=True)

    run_file = config.benchmarks_dir / f"{clean_model_name}.json"
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(full_record, f, indent=2, ensure_ascii=False)

    comparison_file = config.benchmarks_dir / "comparison.json"
    comparison_data: Dict[str, Any] = {}
    if comparison_file.exists():
        try:
            with open(comparison_file, "r", encoding="utf-8") as f:
                comparison_data = json.load(f)
        except Exception:
            comparison_data = {}

    summary = full_record.get("summary", {})
    comparison_data[summary["model"]] = summary

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)


def __load_comparison_records() -> Dict[str, Any]:
    """Loads consolidated comparison.json index."""
    comparison_file = config.benchmarks_dir / "comparison.json"
    if not comparison_file.exists():
        return {}
    try:
        with open(comparison_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------------------------------------------------------------------
# Rich Presentation Tables
# ------------------------------------------------------------------------------

def display_run_summary(run_data: Dict[str, Any], console: Console):
    """Prints a styled Rich Panel summarizing the benchmark run."""
    summary = run_data.get("summary", {})
    if not summary:
        return

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Tested Model", summary.get("model", "Unknown"))
    table.add_row("Evaluation Book", summary.get("book", "Unknown"))
    table.add_row("Questions Tested", str(summary.get("total_questions", 0)))
    table.add_row("Avg TTFT (Time To First Token)", f"{summary.get('avg_ttft_ms', 0)} ms")
    table.add_row("Generation Speed", f"{summary.get('avg_tokens_per_sec', 0)} tokens/sec")
    table.add_row("Avg Total Latency", f"{summary.get('avg_latency_s', 0)} seconds")
    table.add_row("Peak VRAM Usage", f"{summary.get('peak_vram_mb', 0):.0f} MB")
    table.add_row("Chapter Citation Accuracy", f"{summary.get('chapter_accuracy_pct', 0):.1f}%")
    table.add_row("Semantic Similarity", f"{summary.get('avg_semantic_similarity', 0) * 100:.1f}%")

    console.print()
    console.print(Panel(table, title="📊 [bold green]Benchmark Results Summary[/bold green]", border_style="green"))


def display_comparison_table(console: Console):
    """Loads and displays side-by-side comparison across all tested models."""
    records = __load_comparison_records()
    if not records:
        console.print("[yellow]No models benchmarked yet. Run a benchmark first![/yellow]")
        return

    table = Table(
        title="🏆 LLM Model Comparison Matrix",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_blue"
    )
    table.add_column("Model", style="bold white")
    table.add_column("TTFT", justify="right", style="cyan")
    table.add_column("Speed", justify="right", style="green")
    table.add_column("Latency", justify="right", style="yellow")
    table.add_column("Peak VRAM", justify="right", style="red")
    table.add_column("Chapter Acc.", justify="right", style="blue")
    table.add_column("Similarity", justify="right", style="magenta")
    table.add_column("Last Tested", style="dim")

    for model_name, data in records.items():
        dt = data.get("timestamp", "")
        formatted_date = dt.split("T")[0] if "T" in dt else dt

        table.add_row(
            model_name,
            f"{data.get('avg_ttft_ms', 0):.0f} ms",
            f"{data.get('avg_tokens_per_sec', 0):.1f} tok/s",
            f"{data.get('avg_latency_s', 0):.2f} s",
            f"{data.get('peak_vram_mb', 0):.0f} MB",
            f"{data.get('chapter_accuracy_pct', 0):.0f}%",
            f"{data.get('avg_semantic_similarity', 0) * 100:.0f}%",
            formatted_date
        )

    console.print()
    console.print(table)
    console.print(f"Results stored in: {config.benchmarks_dir / 'comparison.json'}\n")
