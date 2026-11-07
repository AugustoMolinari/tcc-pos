import sys
import questionary
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import config
from src.epub_parser import parse_epub
from src.chunker import chunk_book
from src.vector_store import BookVectorStore
from src.llm import OllamaClient
from src.samples import PUBLIC_DOMAIN_BOOKS, download_sample_book
from src.benchmark import (
    run_benchmark_on_model,
    display_run_summary,
    display_comparison_table,
    ALICE_GOLDEN_QUESTIONS
)

console = Console()

MENU_STYLE = questionary.Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold white"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:cyan bold"),
    ("separator", "fg:#6c7086"),
    ("instruction", "fg:#6c7086 italic"),
    ("answer", "fg:green bold"),
])


def select_menu(message: str, choices: list) -> Optional[str]:
    """Helper to display questionary select prompt with custom highlighting."""
    return questionary.select(
        message,
        choices=choices,
        use_indicator=True,
        style=MENU_STYLE
    ).ask()


def get_vector_store() -> BookVectorStore:
    return BookVectorStore(
        persist_dir=config.chroma_dir,
        embedding_model_name=config.embedding_model,
        reranker_model_name=config.reranker_model
    )


def get_llm_client() -> OllamaClient:
    return OllamaClient(
        base_url=config.ollama_url,
        default_model=config.active_model,
        temperature=config.temperature
    )


def resolve_active_model(llm: OllamaClient, model_override: Optional[str] = None) -> str:
    """Returns requested model or gracefully falls back to an available model."""
    if not llm.is_online():
        return model_override or config.active_model

    installed = llm.list_models()
    if not installed:
        return model_override or config.active_model


    if model_override:
        if model_override in installed:
            return model_override

        matching = [m for m in installed if m.startswith(model_override) or model_override in m]
        if matching:
            console.print(f"[cyan]Using matching model: '{matching[0]}' for '{model_override}'[/cyan]")
            return matching[0]

        console.print(f"[yellow]Warning: Requested model '{model_override}' is not installed.[/yellow]")
        console.print(f"[dim]Available local models: {', '.join(installed)}[/dim]")

        candidate = config.active_model if config.active_model in installed else (
            config.fallback_model if config.fallback_model in installed else installed[0]
        )
        console.print(f"[yellow]Falling back to installed model: '{candidate}'[/yellow]\n")
        return candidate

    if config.active_model not in installed:
        if config.fallback_model in installed:
            config.active_model = config.fallback_model
        else:
            config.active_model = installed[0]

    return config.active_model


# ------------------------------------------------------------------------------
# Library & Operations Helpers
# ------------------------------------------------------------------------------

def list_books():
    """List all indexed books in ChromaDB."""
    vs = get_vector_store()
    books = vs.list_indexed_books()
    if not books:
        console.print("[yellow]No books currently indexed. Ingest an EPUB from the menu first.[/yellow]")
        return

    table = Table(title="📚 Indexed Books Library", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Book Title", style="bold white")
    table.add_column("Chapters", justify="right", style="green")
    table.add_column("Chunks", justify="right", style="blue")

    for i, b in enumerate(books, 1):
        table.add_row(
            str(i),
            b["book_title"],
            str(b["chapter_count"]),
            str(b["total_chunks"])
        )

    console.print(table)


def __ingest_single_epub(path: Path) -> bool:
    """Internal helper to parse, chunk, and index an EPUB file."""
    console.print(f"\n[cyan]📖 Reading EPUB:[/cyan] {path.name}")
    try:
        with console.status("[bold green]Parsing chapters and cleaning text..."):
            book_doc = parse_epub(path)
    except Exception as e:
        console.print(f"[bold red]Failed to parse EPUB:[/bold red] {e}")
        return False

    console.print(f"  • [bold white]Title:[/bold white] {book_doc.title}")
    console.print(f"  • [bold white]Chapters extracted:[/bold white] {len(book_doc.chapters)}")
    console.print(f"  • [bold white]Total words:[/bold white] {book_doc.total_words:,}")

    with console.status("[bold green]Generating chapter-aware chunks..."):
        chunks = chunk_book(book_doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    console.print(f"  • [bold white]Chunks created:[/bold white] {len(chunks)}")

    vs = get_vector_store()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Generating embeddings & saving to ChromaDB ({vs.device.upper()})...", total=None)
        added_count = vs.add_book_chunks(chunks)
        progress.update(task, completed=True)

    console.print(f"[bold green]✓ Successfully indexed '{book_doc.title}' ({added_count} chunks)![/bold green]\n")
    return True


# ------------------------------------------------------------------------------
# Interactive Operations
# ------------------------------------------------------------------------------

def __resolve_book_title(input_str: str) -> Optional[str]:
    """Finds exact or closest matching indexed book title."""
    vs = get_vector_store()
    books = vs.list_indexed_books()
    if not books:
        return None

    # Check exact match
    for b in books:
        if b["book_title"].lower() == input_str.lower():
            return b["book_title"]

    # Check substring match
    matches = [b["book_title"] for b in books if input_str.lower() in b["book_title"].lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]Multiple books matched '{input_str}':[/yellow]")
        choices = matches + [questionary.Separator(), "← Cancel"]
        selected = select_menu("Select matched book:", choices=choices)
        if not selected or selected == "← Cancel":
            return None
        return selected

    return None


def ask_scene_question(book_title: str, question: str, model_override: Optional[str] = None):
    """Executes a retrieval and streams an answer with scene context & chapter citations."""
    resolved_title = __resolve_book_title(book_title)
    if not resolved_title:
        console.print(f"[bold red]Book not found in database:[/bold red] '{book_title}'")
        console.print("[dim]Use 'book-rag list' to view indexed titles.[/dim]")
        return

    vs = get_vector_store()
    llm = get_llm_client()

    if not llm.is_online():
        console.print(f"[bold red]Error: Ollama is offline at {llm.base_url}[/bold red]")
        console.print("[dim]Please start Ollama with 'ollama serve' in another terminal.[/dim]")
        return

    active_model = resolve_active_model(llm, model_override)

    console.print(f"\n[cyan]🔍 Searching '{resolved_title}' for relevant scenes...[/cyan]")
    chunks = vs.search(query=question, top_k=config.top_k, fetch_k=config.fetch_k, book_title=resolved_title)

    if not chunks:
        console.print("[yellow]No relevant excerpts found for your query.[/yellow]")
        return

    # Print citation overview
    cite_table = Table(title="📌 Retrieved Scene Excerpts", show_header=True, header_style="bold magenta")
    cite_table.add_column("Excerpt", style="dim", width=8)
    cite_table.add_column("Chapter / Section", style="bold cyan")
    cite_table.add_column("Score", justify="right", style="green")
    cite_table.add_column("Excerpt Preview", style="white")

    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ")[:90] + "..."
        cite_table.add_row(
            f"#{i}",
            c["chapter_title"],
            f"{c["score"]:.3f}",
            preview
        )
    console.print(cite_table)

    # Prompt generation & streaming
    console.print(f"\n[bold green]🤖 Answer ({active_model}):[/bold green]")
    try:
        for token in llm.stream_chat(query=question, book_title=resolved_title, retrieved_chunks=chunks, model=active_model):
            console.print(token, end="", style="bold white")
            sys.stdout.flush()
        console.print("\n")
    except Exception as e:
        console.print(f"\n[bold red]Generation error:[/bold red] {e}")


def interactive_chat_session(book_title: str, model_override: Optional[str] = None):
    """Enters an ongoing interactive chat session with a book."""
    resolved_title = __resolve_book_title(book_title)
    if not resolved_title:
        console.print(f"[bold red]Book not found in database:[/bold red] '{book_title}'")
        return

    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        return

    active_model = resolve_active_model(llm, model_override)
    console.print(Panel(
        f"[bold cyan]Chatting with:[/bold cyan] [bold white]{resolved_title}[/bold white]\n"
        f"[bold cyan]Active Model:[/bold cyan] {active_model}\n\n"
        f"[dim]Type your questions below. Type [bold red]'exit'[/bold red] or [bold red]'back'[/bold red] to return to the menu.[/dim]",
        title="💬 Book Q&A Session",
        border_style="cyan"
    ))

    while True:
        try:
            query = Prompt.ask("\n[bold yellow]You[/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            break

        if not query.strip():
            continue
        if query.strip().lower() in ["exit", "quit", "back"]:
            break

        ask_scene_question(resolved_title, query, model_override=active_model)


def ingest_flow() -> bool:
    """Interactive book scanner & selective ingest."""
    default_dir = str(config.books_dir)
    dir_input = questionary.text(
        "Enter directory to scan for .epub files (or press Enter for default):",
        default=default_dir,
        style=MENU_STYLE
    ).ask()
    if dir_input is None:
        console.print("[dim]Ingest cancelled.[/dim]")
        return False

    scan_path = Path(dir_input.strip()).expanduser().resolve()
    if not scan_path.exists() or not scan_path.is_dir():
        console.print(f"[bold red]Directory not found:[/bold red] {scan_path}")
        return False

    epub_files = sorted(list(scan_path.glob("*.epub")) + list(scan_path.glob("*/*.epub")))
    if not epub_files:
        console.print(f"[yellow]No .epub files found in {scan_path}[/yellow]")
        return False

    table = Table(title=f"📁 EPUB Files Found in {scan_path.name}", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("File Name", style="bold white")
    table.add_column("Size", justify="right", style="green")

    for i, ep in enumerate(epub_files, 1):
        size_mb = ep.stat().st_size / (1024 * 1024)
        table.add_row(str(i), ep.name, f"{size_mb:.1f} MB")

    console.print(table)

    file_options = [
        f"All files ({len(epub_files)} books)",
    ] + [
        f"{ep.name} ({ep.stat().st_size / (1024 * 1024):.1f} MB)"
        for ep in epub_files
    ] + [
        questionary.Separator(),
        "← Cancel / Back to Main Menu"
    ]

    chosen = select_menu("Select EPUB file(s) to ingest:", choices=file_options)

    if not chosen or chosen == "← Cancel / Back to Main Menu":
        console.print("[dim]Ingest cancelled.[/dim]")
        return False

    if chosen.startswith("All files"):
        selected_files = epub_files
    else:
        file_idx = [f"{ep.name} ({ep.stat().st_size / (1024 * 1024):.1f} MB)" for ep in epub_files].index(chosen)
        selected_files = [epub_files[file_idx]]

    for ep in selected_files:
        __ingest_single_epub(ep)
    return True


def delete_book_flow() -> bool:
    """Interactive flow to delete an indexed book from ChromaDB."""
    vs = get_vector_store()
    books = vs.list_indexed_books()
    if not books:
        console.print("[yellow]No books currently indexed. Nothing to delete.[/yellow]")
        return False

    table = Table(title="🗑️  Delete Book from Vector Store", show_header=True, header_style="bold red")
    table.add_column("#", style="dim", width=4)
    table.add_column("Book Title", style="bold white")
    table.add_column("Chapters", justify="right", style="green")
    table.add_column("Chunks", justify="right", style="blue")

    for i, b in enumerate(books, 1):
        table.add_row(
            str(i),
            b["book_title"],
            str(b["chapter_count"]),
            str(b["total_chunks"])
        )

    console.print(table)

    book_options = [
        f"{b['book_title']} ({b['chapter_count']} ch, {b['total_chunks']} chunks)"
        for b in books
    ] + [
        questionary.Separator(),
        "← Cancel / Back to Main Menu"
    ]

    selected_choice = select_menu("Select book to delete from ChromaDB:", choices=book_options)

    if not selected_choice or selected_choice == "← Cancel / Back to Main Menu":
        console.print("[dim]Deletion cancelled.[/dim]\n")
        return False

    idx = [f"{b['book_title']} ({b['chapter_count']} ch, {b['total_chunks']} chunks)" for b in books].index(selected_choice)
    selected_book = books[idx]
    title = selected_book["book_title"]
    chunks_count = selected_book["total_chunks"]

    confirmed = questionary.confirm(
        f"Are you sure you want to permanently delete '{title}' ({chunks_count} chunks)?",
        default=False,
        style=MENU_STYLE
    ).ask()

    if confirmed:
        with console.status(f"[bold red]Deleting '{title}' from ChromaDB...[/bold red]"):
            deleted = vs.delete_book(title)
        console.print(f"[bold green]✓ Successfully removed '{title}' ({deleted} chunks deleted)![/bold green]\n")
        return True
    else:
        console.print("[yellow]Deletion cancelled.[/yellow]\n")
        return False


def switch_model_flow() -> bool:
    """Interactive selector to switch the active Ollama model."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        return False

    models = llm.list_models()
    if not models:
        console.print("[yellow]No local models found in Ollama.[/yellow]")
        return False

    model_choices = [
        f"{m} (current)" if m == config.active_model else m
        for m in models
    ] + [
        questionary.Separator(),
        "← Back to Main Menu"
    ]

    choice = select_menu("Select Ollama model to activate:", choices=model_choices)

    if not choice or choice == "← Back to Main Menu":
        return False

    chosen_model = choice.replace(" (current)", "").strip()
    config.active_model = chosen_model
    console.print(f"[bold green]✓ Active model set to: {config.active_model}[/bold green]\n")
    return False


def pull_model_flow() -> bool:
    """Interactive prompt to pull a new model from Ollama."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        return False

    rec_choices = [
        "qwen2.5:7b — Best comprehension, reasoning",
        "llama3.2:3b — Blazing fast, ultra-lightweight",
        "Custom model name (type manually)...",
        questionary.Separator(),
        "← Cancel / Back to Main Menu"
    ]

    choice = select_menu("Choose a model to pull via Ollama:", choices=rec_choices)

    if not choice or choice == "← Cancel / Back to Main Menu":
        console.print("[dim]Pull cancelled.[/dim]\n")
        return False

    if choice.startswith("Custom model name"):
        model_name = questionary.text("Enter model name to pull (e.g. gemma2:2b):", style=MENU_STYLE).ask()
        if not model_name or not model_name.strip():
            console.print("[dim]Pull cancelled.[/dim]\n")
            return False
        model_name = model_name.strip()
    else:
        model_name = choice.split(" — ")[0].strip()

    console.print(f"[cyan]Downloading and preparing '{model_name}' via Ollama...[/cyan]")

    try:
        for update in llm.pull_model(model_name):
            status = update.get("status", "")
            completed = update.get("completed", 0)
            total = update.get("total", 0)
            if total > 0:
                pct = (completed / total) * 100
                console.print(f"\r  {status}: {pct:.1f}%", end="")
            else:
                console.print(f"\r  {status}", end="")
            sys.stdout.flush()
        console.print(f"\n[bold green]✓ Model '{model_name}' ready to use![/bold green]\n")
        config.active_model = model_name
        return True
    except Exception as e:
        console.print(f"\n[bold red]Failed to pull model:[/bold red] {e}\n")
        return True


def download_public_domain_books_flow() -> bool:
    """Downloads classic public domain books from Project Gutenberg."""
    keys = list(PUBLIC_DOMAIN_BOOKS.keys())
    book_choices = [
        f"{PUBLIC_DOMAIN_BOOKS[k]['title']} — by {PUBLIC_DOMAIN_BOOKS[k]['author']}"
        for k in keys
    ] + [
        questionary.Separator(),
        "← Cancel / Back to Main Menu"
    ]

    choice = select_menu("Select a classic book to download from Project Gutenberg:", choices=book_choices)

    if not choice or choice == "← Cancel / Back to Main Menu":
        console.print("[dim]Download cancelled.[/dim]\n")
        return False

    idx = [f"{PUBLIC_DOMAIN_BOOKS[k]['title']} — by {PUBLIC_DOMAIN_BOOKS[k]['author']}" for k in keys].index(choice)
    chosen_key = keys[idx]
    info = PUBLIC_DOMAIN_BOOKS[chosen_key]

    with console.status(f"[bold green]Downloading '{info['title']}' from Project Gutenberg..."):
        try:
            epub_path = download_sample_book(chosen_key, config.samples_dir)
            console.print(f"[bold green]✓ Downloaded to:[/bold green] {epub_path}")
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}\n")
            return True

    ingest_now = questionary.confirm("Would you like to ingest this book right now?", default=True, style=MENU_STYLE).ask()
    if ingest_now:
        __ingest_single_epub(epub_path)
    return True


def benchmark_flow() -> bool:
    """Interactive flow to benchmark models against Alice in Wonderland and compare metrics."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        console.print("[dim]Start Ollama with 'ollama serve' first.[/dim]")
        return True

    active_m = resolve_active_model(llm)

    console.print("\n[bold cyan]Model Benchmarking & Metrics Suite:[/bold cyan]")
    console.print(f"  Active Model: [bold green]{active_m}[/bold green]")
    console.print(f"  Target Book: [bold white]Alice's Adventures in Wonderland[/bold white] ({len(ALICE_GOLDEN_QUESTIONS)} Golden Questions)")
    console.print("  Metrics: [dim]TTFT (ms), Tokens/sec, Total Latency, Peak VRAM, Chapter Citation Accuracy, Semantic Similarity[/dim]\n")

    sub_choice = select_menu(
        "Choose benchmark action:",
        choices=[
            questionary.Choice("Run Benchmark on Active Model", value="run"),
            questionary.Choice("View Model Comparison (All saved benchmarks)", value="compare"),
            questionary.Separator(),
            questionary.Choice("← Back to Main Menu", value="back"),
        ]
    )

    if not sub_choice or sub_choice == "back":
        return False
    elif sub_choice == "compare":
        display_comparison_table(console)
        return True

    result = run_benchmark_on_model(
        model_name=active_m,
        questions=ALICE_GOLDEN_QUESTIONS,
        console=console
    )

    if result:
        display_run_summary(result, console)
        display_comparison_table(console)
        return True
    return False


# ------------------------------------------------------------------------------
# Default Interactive Menu Loop
# ------------------------------------------------------------------------------

def run_interactive_menu():
    """Main menu loop when book-rag is run without CLI arguments."""
    llm = get_llm_client()
    vs = get_vector_store()

    while True:
        console.clear()

        # Detect status
        is_online = llm.is_online()
        ollama_status = "[bold green]Online[/bold green]" if is_online else "[bold red]Offline (Run 'ollama serve')[/bold red]"

        # Graceful model fallback if active model is not installed
        if is_online:
            config.active_model = resolve_active_model(llm)

        books = vs.list_indexed_books()
        book_count = len(books)

        console.print()
        console.print(Panel(
            f" [bold white]Ollama Engine:[/bold white] {ollama_status}   |   "
            f"[bold white]Active Model:[/bold white] [cyan]{config.active_model}[/cyan]   |   "
            f"[bold white]Books Indexed:[/bold white] [magenta]{book_count}[/magenta]",
            title="📚 [bold cyan]Book RAG — Local EPUB Assistant[/bold cyan] 📚",
            subtitle="Local & Offline Literary Exploration",
            border_style="bright_blue"
        ))

        menu_choices = [
            questionary.Choice("Ingest Book from folder (select .epub)", value="ingest"),
            questionary.Choice("Chat about a Book (interactive conversation)", value="chat"),
            questionary.Choice("Ask a Scene Question (single question)", value="ask"),
            questionary.Choice("List Indexed Books & Library", value="list"),
            questionary.Choice("Delete a Book from Vector Store", value="delete"),
            questionary.Choice("Switch Active LLM Model", value="switch_model"),
            questionary.Choice("Download / Pull a New Model via Ollama", value="pull_model"),
            questionary.Choice("Download Public Domain Books", value="download_samples"),
            questionary.Choice("Benchmark & Compare Models", value="benchmark"),
            questionary.Separator(),
            questionary.Choice("Exit", value="exit"),
        ]

        action = select_menu("Main Menu (Use ↑/↓ arrows to navigate, Enter to select):", choices=menu_choices)
        performed_action = False

        if not action or action == "exit":
            console.print("[cyan]Goodbye! Happy reading![/cyan]")
            break
        elif action == "ingest":
            performed_action = ingest_flow()
        elif action == "chat":
            if not books:
                console.print("[yellow]No books indexed yet! Ingest a book first.[/yellow]")
                continue
            book_choices = [b["book_title"] for b in books] + [
                questionary.Separator(),
                "← Back to Main Menu"
            ]
            selected_title = select_menu("Select book to chat with:", choices=book_choices)
            if not selected_title or selected_title == "← Back to Main Menu":
                continue
            interactive_chat_session(selected_title)
            performed_action = True
        elif action == "ask":
            if not books:
                console.print("[yellow]No books indexed yet! Ingest a book first.[/yellow]")
                continue
            book_choices = [b["book_title"] for b in books] + [
                questionary.Separator(),
                "← Back to Main Menu"
            ]
            selected_title = select_menu("Select book to query:", choices=book_choices)
            if not selected_title or selected_title == "← Back to Main Menu":
                continue
            query = Prompt.ask(f"Enter question about '{selected_title}' (or press Enter to cancel)")
            if query.strip() and query.strip().lower() not in ["back", "cancel", "exit"]:
                ask_scene_question(selected_title, query)
                performed_action = True
            else:
                continue
        elif action == "list":
            list_books()
            performed_action = True
        elif action == "delete":
            performed_action = delete_book_flow()
        elif action == "switch_model":
            performed_action = switch_model_flow()
        elif action == "pull_model":
            performed_action = pull_model_flow()
        elif action == "download_samples":
            performed_action = download_public_domain_books_flow()
        elif action == "benchmark":
            performed_action = benchmark_flow()

        if performed_action:
            Prompt.ask("\n[dim]Press Enter to return to the main menu...[/dim]", default="", show_default=False)


def main():
    """Launch the interactive Book RAG assistant."""
    run_interactive_menu()


if __name__ == "__main__":
    main()
