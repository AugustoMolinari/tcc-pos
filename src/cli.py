import sys
import typer
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

console = Console()
app = typer.Typer(
    help="📚 Book RAG — Local EPUB Q&A Assistant",
    no_args_is_help=False
)


def get_vector_store() -> BookVectorStore:
    return BookVectorStore(
        persist_dir=config.chroma_dir,
        embedding_model_name=config.embedding_model
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
# CLI Direct Commands
# ------------------------------------------------------------------------------

@app.command("list")
def list_books():
    """List all indexed books in ChromaDB."""
    vs = get_vector_store()
    books = vs.list_indexed_books()
    if not books:
        console.print("[yellow]No books currently indexed. Run 'book-rag ingest' or choose option 1 in the menu.[/yellow]")
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


@app.command("ingest")
def ingest_command(
    epub_path: str = typer.Argument(..., help="Path to the .epub file to ingest")
):
    """Parse and index a single EPUB file."""
    path = Path(epub_path).expanduser().resolve()
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(1)

    __ingest_single_epub(path)


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


@app.command("ask")
def ask_command(
    book_title: str = typer.Argument(..., help="Title (or part of title) of the indexed book"),
    question: str = typer.Argument(..., help="Question to ask about the book"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model to use")
):
    """Ask a single question about an indexed book."""
    ask_scene_question(book_title, question, model_override=model)


@app.command("chat")
def chat_command(
    book_title: str = typer.Argument(..., help="Title (or part of title) of the indexed book"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model to use")
):
    """Start an interactive chat session with an indexed book."""
    interactive_chat_session(book_title, model_override=model)


@app.command("models")
def models_command():
    """List locally installed Ollama models."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline or unreachable at {llm.base_url}[/bold red]")
        console.print("[dim]Run 'ollama serve' in a terminal to start the Ollama daemon.[/dim]")
        return

    models = llm.list_models()
    if not models:
        console.print("[yellow]No models found in local Ollama. Pull one with: ollama pull llama3.2:3b[/yellow]")
        return

    table = Table(title="🤖 Installed Ollama Models", show_header=True, header_style="bold green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model Name", style="bold white")
    table.add_column("Active", justify="center")

    for i, m in enumerate(models, 1):
        is_active = "✓ (Current)" if m == config.active_model else ""
        table.add_row(str(i), m, f"[bold green]{is_active}[/bold green]" if is_active else "")

    console.print(table)


@app.command("download-samples")
def download_samples_command():
    """Download public domain sample books from Project Gutenberg."""
    download_public_domain_books_flow()


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
        for i, m in enumerate(matches, 1):
            console.print(f"  [{i}] {m}")
        choice = Prompt.ask("Select book number", choices=[str(i) for i in range(1, len(matches) + 1)])
        return matches[int(choice) - 1]

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
    chunks = vs.search(query=question, top_k=config.top_k, book_title=resolved_title)

    if not chunks:
        console.print("[yellow]No relevant excerpts found for your query.[/yellow]")
        return

    # Print citation overview
    cite_table = Table(title="📌 Retrieved Scene Excerpts", show_header=True, header_style="bold magenta")
    cite_table.add_column("Excerpt", style="dim", width=8)
    cite_table.add_column("Chapter / Section", style="bold cyan")
    cite_table.add_column("Similarity", justify="right", style="green")
    cite_table.add_column("Excerpt Preview", style="white")

    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ")[:90] + "..."
        cite_table.add_row(
            f"#{i}",
            c["chapter_title"],
            f"{c['score']:.2f}",
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


def ingest_flow():
    """Interactive book scanner & selective ingest."""
    default_dir = str(config.books_dir)
    dir_input = Prompt.ask(
        "Enter directory to scan for .epub files",
        default=default_dir
    )
    scan_path = Path(dir_input).expanduser().resolve()
    if not scan_path.exists() or not scan_path.is_dir():
        console.print(f"[bold red]Directory not found:[/bold red] {scan_path}")
        return

    epub_files = sorted(list(scan_path.glob("*.epub")) + list(scan_path.glob("*/*.epub")))
    if not epub_files:
        console.print(f"[yellow]No .epub files found in {scan_path}[/yellow]")
        return

    table = Table(title=f"📁 EPUB Files Found in {scan_path.name}", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("File Name", style="bold white")
    table.add_column("Size", justify="right", style="green")

    for i, ep in enumerate(epub_files, 1):
        size_mb = ep.stat().st_size / (1024 * 1024)
        table.add_row(str(i), ep.name, f"{size_mb:.1f} MB")

    console.print(table)

    selection = Prompt.ask(
        "\nSelect file number to ingest (e.g. '1', or '1,3', or 'all', or '0' to cancel)",
        default="1"
    )

    if selection.strip() == "0":
        return

    if selection.strip().lower() == "all":
        selected_files = epub_files
    else:
        try:
            indices = [int(s.strip()) - 1 for s in selection.split(",") if s.strip()]
            selected_files = [epub_files[idx] for idx in indices if 0 <= idx < len(epub_files)]
        except Exception:
            console.print("[bold red]Invalid selection.[/bold red]")
            return

    for ep in selected_files:
        __ingest_single_epub(ep)


def switch_model_flow():
    """Interactive selector to switch the active Ollama model."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        return

    models = llm.list_models()
    if not models:
        console.print("[yellow]No local models found in Ollama.[/yellow]")
        return

    console.print("\n[bold cyan]Available Local Models:[/bold cyan]")
    for i, m in enumerate(models, 1):
        marker = " [bold green](current)[/bold green]" if m == config.active_model else ""
        console.print(f"  [{i}] {m}{marker}")

    choice = Prompt.ask("Select model number to activate", choices=[str(i) for i in range(1, len(models) + 1)])
    config.active_model = models[int(choice) - 1]
    console.print(f"[bold green]✓ Active model set to: {config.active_model}[/bold green]")


def pull_model_flow():
    """Interactive prompt to pull a new model from Ollama."""
    llm = get_llm_client()
    if not llm.is_online():
        console.print(f"[bold red]Ollama is offline at {llm.base_url}[/bold red]")
        return

    console.print("\n[bold cyan]Recommended Models:[/bold cyan]")
    console.print("  • [bold white]qwen2.5:7b[/bold white] (Best comprehension, reasoning & multilingual support)")
    console.print("  • [bold white]llama3.2:3b[/bold white] (Blazing fast, ultra-lightweight ~2.2GB VRAM)")
    console.print("  • [bold white]mistral:7b[/bold white] (Solid general storytelling & reasoning)")
    console.print("  • [bold white]phi3.5:3.8b[/bold white] (High quality compact model)")
    console.print("  • [bold white]deepseek-r1:7b[/bold white] (Reasoning-focused model)")

    model_name = Prompt.ask("\nEnter model name to pull", default="qwen2.5:7b")
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
        console.print(f"\n[bold green]✓ Model '{model_name}' ready to use![/bold green]")
        config.active_model = model_name
    except Exception as e:
        console.print(f"\n[bold red]Failed to pull model:[/bold red] {e}")


def download_public_domain_books_flow():
    """Downloads classic public domain books from Project Gutenberg."""
    console.print("\n[bold cyan]Classic Public Domain Books:[/bold cyan]")
    keys = list(PUBLIC_DOMAIN_BOOKS.keys())
    for i, k in enumerate(keys, 1):
        info = PUBLIC_DOMAIN_BOOKS[k]
        console.print(f"  [{i}] [bold white]{info['title']}[/bold white] by {info['author']}")

    choice = Prompt.ask("\nSelect book number to download", choices=[str(i) for i in range(1, len(keys) + 1)])
    chosen_key = keys[int(choice) - 1]
    info = PUBLIC_DOMAIN_BOOKS[chosen_key]

    with console.status(f"[bold green]Downloading '{info['title']}' from Project Gutenberg..."):
        try:
            epub_path = download_sample_book(chosen_key, config.samples_dir)
            console.print(f"[bold green]✓ Downloaded to:[/bold green] {epub_path}")
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}")
            return

    if Confirm.ask("Would you like to ingest this book right now?", default=True):
        __ingest_single_epub(epub_path)


# ------------------------------------------------------------------------------
# Default Interactive Menu Loop
# ------------------------------------------------------------------------------

def run_interactive_menu():
    """Main menu loop when book-rag is run without CLI arguments."""
    llm = get_llm_client()
    vs = get_vector_store()

    while True:
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

        console.print("[bold white]Main Menu:[/bold white]")
        console.print("  [bold green][1][/bold green] Ingest Book from folder (select .epub)")
        console.print("  [bold green][2][/bold green] Chat with a Book (interactive conversation)")
        console.print("  [bold green][3][/bold green] Ask a Scene Question (with chapter citations)")
        console.print("  [bold green][4][/bold green] List Indexed Books & Library")
        console.print("  [bold green][5][/bold green] Switch Active LLM Model")
        console.print("  [bold green][6][/bold green] Download / Pull a New Model via Ollama")
        console.print("  [bold green][7][/bold green] Download Public Domain Books (Alice in Wonderland, etc.)")
        console.print("  [bold red][0][/bold red] Exit")

        choice = Prompt.ask("\nChoose an option", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")

        if choice == "0":
            console.print("[cyan]Goodbye! Happy reading![/cyan]")
            break
        elif choice == "1":
            ingest_flow()
        elif choice == "2":
            if not books:
                console.print("[yellow]No books indexed yet! Ingest a book first (Option 1).[/yellow]")
                continue
            console.print("\n[bold cyan]Indexed Books:[/bold cyan]")
            for i, b in enumerate(books, 1):
                console.print(f"  [{i}] {b['book_title']}")
            b_choice = Prompt.ask("Select book number", choices=[str(i) for i in range(1, len(books) + 1)])
            selected_title = books[int(b_choice) - 1]["book_title"]
            interactive_chat_session(selected_title)
        elif choice == "3":
            if not books:
                console.print("[yellow]No books indexed yet! Ingest a book first (Option 1).[/yellow]")
                continue
            console.print("\n[bold cyan]Indexed Books:[/bold cyan]")
            for i, b in enumerate(books, 1):
                console.print(f"  [{i}] {b['book_title']}")
            b_choice = Prompt.ask("Select book number", choices=[str(i) for i in range(1, len(books) + 1)])
            selected_title = books[int(b_choice) - 1]["book_title"]
            query = Prompt.ask(f"Enter question about '{selected_title}'")
            if query.strip():
                ask_scene_question(selected_title, query)
        elif choice == "4":
            list_books()
        elif choice == "5":
            switch_model_flow()
        elif choice == "6":
            pull_model_flow()
        elif choice == "7":
            download_public_domain_books_flow()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Default callback: If no subcommands are provided, launch the interactive menu."""
    if ctx.invoked_subcommand is None:
        run_interactive_menu()


if __name__ == "__main__":
    app()
