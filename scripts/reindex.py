"""
reindex.py
──────────


What it does:
    1. finds the document(s) to reindex
    2. deletes existing chunks from Supabase
    3. loads fresh content from PDF
    4. chunks and embeds the fresh content
    5. stores new chunks in Supabase
    6. logs everything to MLflow

"""

 
import argparse
import logging
import sys
import time
from pathlib import Path
 
import mlflow
from rich.console import Console
from rich.table import Table
 
# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from config import settings
from craster_rag.ingestion.loader_factory import LoadFactory
from craster_rag.ingestion.chunker import Chunker
from craster_rag.ingestion.embedder import Embedder
from craster_rag.retrieval.vector_store import VectorStore
from craster_rag.cache.cache_client import cache
 
# ── Logging setup ──────────────────────────────────────
logging.basicConfig(
    level  = getattr(logging, settings.log_level),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger  = logging.getLogger(__name__)
console = Console()
 
# ── Documents path ─────────────────────────────────────
DOCUMENTS_PATH = Path("data/documents/hr_policy")
 
 
def reindex_file(filename: str) -> dict:
    """
    Reindex a single PDF file.
 
    Steps:
        1. find the file in hr_policy folder
        2. delete existing chunks for this file
        3. load fresh content
        4. chunk embed and store
 
    Args:
        filename: PDF filename e.g. "Maternity Policy.pdf"
 
    Returns:
        dict with reindex statistics
    """
    start_time = time.time()
 
    console.print(
        f"\n[bold green]Reindexing:[/bold green] [cyan]{filename}[/cyan]\n"
    )
 
    # find the file
    file_path = DOCUMENTS_PATH / filename
    if not file_path.exists():
        console.print(f"[bold red]File not found: {file_path}[/bold red]")
        return {}
 
    store = VectorStore()
 
    # step 1 — delete existing chunks
    console.print("[bold]Step 1/4[/bold] Deleting existing chunks...")
    deleted = store.delete_by_source(str(file_path.resolve()))
    console.print(
        f"         Deleted [yellow]{deleted}[/yellow] old chunk(s)\n"
    )
 
    # step 2 — load fresh content
    console.print("[bold]Step 2/4[/bold] Loading fresh content...")
    loader    = LoadFactory.get_loader(".pdf")
    documents = loader.load(str(file_path))
 
    if not documents:
        console.print("[bold red]No content loaded. Check the file.[/bold red]")
        return {}
 
    console.print(
        f"         Loaded [green]{len(documents)}[/green] page(s)\n"
    )
 
    # step 3 — chunk
    console.print("[bold]Step 3/4[/bold] Chunking...")
    chunker = Chunker(
        chunk_size    = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    console.print(
        f"         Created [green]{len(chunks)}[/green] chunk(s)\n"
    )
 
    # step 4 — embed and store
    console.print("[bold]Step 4/4[/bold] Embedding and storing...")
    embedder        = Embedder(
        model_name = settings.embedding_model,
        batch_size = settings.embedding_batch_size,
    )
    embedded_chunks = embedder.embed_chunks(chunks)
    stored          = store.add_chunks(
        embedded_chunks,
        skip_existing=False,   # force store even if exists
    )
 
    total_time = round(time.time() - start_time, 2)
 
    stats = {
        "filename"           : filename,
        "pages_loaded"       : len(documents),
        "chunks_created"     : len(chunks),
        "chunks_stored"      : stored,
        "chunks_deleted"     : deleted,
        "total_time_seconds" : total_time,
    }
 
    _print_file_summary(stats)
    return stats
 
 
def reindex_all() -> dict:
    """
    Reindex all PDF files in hr_policy folder.
 
    Deletes all existing chunks first then
    re-indexes everything fresh.
 
    Returns:
        dict with overall reindex statistics
    """
    start_time = time.time()
 
    console.print(
        "\n[bold green]Reindexing All Documents[/bold green]\n"
    )
 
    store = VectorStore()
 
    # get current stats before deletion
    try:
        before_stats = store.get_stats()
        total_before = before_stats.get("total_chunks", 0)
    except Exception:
        total_before = 0
 
    # find all PDFs
    pdf_files = sorted(DOCUMENTS_PATH.glob("*.pdf"))
 
    if not pdf_files:
        console.print(
            f"[bold red]No PDFs found in {DOCUMENTS_PATH}[/bold red]"
        )
        return {}
 
    console.print(
        f"Found [green]{len(pdf_files)}[/green] PDF(s) to reindex\n"
    )
 
    # step 1 — delete all existing chunks
    console.print("[bold]Step 1/4[/bold] Deleting all existing chunks...")
    total_deleted = 0
    for pdf_file in pdf_files:
        deleted = store.delete_by_source(str(pdf_file.resolve()))
        total_deleted += deleted
 
    console.print(
        f"         Deleted [yellow]{total_deleted}[/yellow] old chunk(s)\n"
    )
 
    # step 2 — load all fresh content
    console.print("[bold]Step 2/4[/bold] Loading all documents...")
    documents = LoadFactory.load_all(str(DOCUMENTS_PATH))
 
    if not documents:
        console.print("[bold red]No documents loaded.[/bold red]")
        return {}
 
    console.print(
        f"         Loaded [green]{len(documents)}[/green] page(s) "
        f"from [green]{len(pdf_files)}[/green] PDF(s)\n"
    )
 
    # step 3 — chunk all
    console.print("[bold]Step 3/4[/bold] Chunking all documents...")
    chunker = Chunker(
        chunk_size    = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    console.print(
        f"         Created [green]{len(chunks)}[/green] chunk(s)\n"
    )
 
    # step 4 — embed and store all
    console.print("[bold]Step 4/4[/bold] Embedding and storing...")
    embedder        = Embedder(
        model_name = settings.embedding_model,
        batch_size = settings.embedding_batch_size,
    )
    embedded_chunks = embedder.embed_chunks(chunks)
    stored          = store.add_chunks(
        embedded_chunks,
        skip_existing=False,   # force store
    )
 
    total_time = round(time.time() - start_time, 2)
 
    stats = {
        "files_reindexed"    : len(pdf_files),
        "pages_loaded"       : len(documents),
        "chunks_created"     : len(chunks),
        "chunks_stored"      : stored,
        "chunks_deleted"     : total_deleted,
        "chunks_before"      : total_before,
        "total_time_seconds" : total_time,
    }
 
    _print_all_summary(stats, store)
    return stats
 
 
def _print_file_summary(stats: dict) -> None:
    """Print summary table for single file reindex."""
    console.print(
        f"\n[bold green]Reindex Complete:[/bold green] "
        f"{stats['filename']}\n"
    )
 
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="dim")
    table.add_column("Value",   justify="right")
 
    table.add_row("Pages loaded",    str(stats["pages_loaded"]))
    table.add_row("Chunks deleted",  str(stats["chunks_deleted"]))
    table.add_row("Chunks created",  str(stats["chunks_created"]))
    table.add_row("Chunks stored",   str(stats["chunks_stored"]))
    table.add_row("Total time",      f"{stats['total_time_seconds']}s")
 
    console.print(table)
 
 
def _print_all_summary(stats: dict, store: VectorStore) -> None:
    """Print summary table for full reindex."""
    console.print("\n[bold green]Full Reindex Complete![/bold green]\n")
 
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="dim")
    table.add_column("Value",   justify="right")
 
    table.add_row("Files reindexed",  str(stats["files_reindexed"]))
    table.add_row("Pages loaded",     str(stats["pages_loaded"]))
    table.add_row("Chunks deleted",   str(stats["chunks_deleted"]))
    table.add_row("Chunks created",   str(stats["chunks_created"]))
    table.add_row("Chunks stored",    str(stats["chunks_stored"]))
    table.add_row("Total time",       f"{stats['total_time_seconds']}s")
 
    console.print(table)
 
    # show category breakdown
    try:
        db_stats = store.get_stats()
        console.print(
            f"\n[dim]Supabase total: "
            f"{db_stats['total_chunks']} chunks[/dim]"
        )
        if db_stats.get("categories"):
            console.print("\n[dim]Category breakdown:[/dim]")
            for cat, count in db_stats["categories"].items():
                console.print(f"  [dim]{cat}: {count} chunks[/dim]")
    except Exception:
        pass
 
 
def main():
    """
    Entry point with argument parsing.
 
    Parses --file or --all argument.
    Wraps reindex in MLflow run.
    """
    parser = argparse.ArgumentParser(
        description="Reindex policy documents in Supabase"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type    = str,
        help    = "Filename to reindex e.g. 'Maternity Policy.pdf'",
    )
    group.add_argument(
        "--all",
        action  = "store_true",
        help    = "Reindex all documents",
    )
    args = parser.parse_args()
 
    # MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
 
    with mlflow.start_run(run_name="reindex"):
        mlflow.log_params({
            "chunk_size"      : settings.chunk_size,
            "chunk_overlap"   : settings.chunk_overlap,
            "embedding_model" : settings.embedding_model,
            "mode"            : "file" if args.file else "all",
            "target"          : args.file or "all",
        })
 
        if args.file:
            stats = reindex_file(args.file)
        else:
            stats = reindex_all()
 
        if not stats:
            console.print("[bold red]Reindex failed.[/bold red]")
            sys.exit(1)
 
        # log metrics
        mlflow.log_metrics({
            k: v for k, v in stats.items()
            if isinstance(v, (int, float))
        })
 
        # ── Invalidate cache ─────────────────────────────
        # previously cached answers may be stale or
        # incomplete relative to the freshly indexed content
        cleared = cache.invalidate_all()
        if cleared > 0:
            console.print(
                f"[dim]Cache invalidated: {cleared} cached "
                f"answer(s) cleared[/dim]"
            )
 
 
if __name__ == "__main__":
    main()
 