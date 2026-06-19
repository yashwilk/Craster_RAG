"""
What it does:
    1. starts MLflow run for experiment tracking
    2. loads .txt files from procedures folder
    3. chunks each document
    4. embeds each chunk
    5. stores in Supabase
    6. logs all metrics to MLflow
"""


import logging
import sys
import time
from pathlib import Path
import mlflow
from rich.console import Console
from rich.table import Table

# add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, DOCUMENT_CATEGORIES
from craster_rag.ingestion.loader_factory import LoaderFactory
from craster_rag.ingestion.chunker import Chunker
from craster_rag.ingestion.embedder import Embedder
from craster_rag.retrieval.vector_store import VectorStore
 
logging.basicConfig(
    level  = getattr(logging, settings.log_level),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger  = logging.getLogger(__name__)
console = Console()


def run_ingestion(documents_path: str = "data/documents/procedures") -> dict:
    """ 
    Steps:
        1. Load all PDFs from documents_path
        2. Chunk each document page
        3. Embed all chunks
        4. Store in Supabase
 """
    start_time = time.time()
    console.print("\n[bold green]Craster RAG — Ingestion Pipeline[/bold green]")
    console.print(f"Source: [cyan]{documents_path}[/cyan]\n")

    # ── Step 1: Load documents ───────────────────────
    console.print("[bold]Step 1/4[/bold] Loading PDFs...")
 
    try:
        documents = LoaderFactory.load_all(documents_path)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return {}
 
    if not documents:
        console.print(
            "[bold red]No documents found. "
            "Check your documents_path.[/bold red]"
        )
        return {}
 
    console.print(
        f"         Loaded [green]{len(documents)}[/green] "
        f"page(s) from "
        f"[green]{_count_unique_files(documents)}[/green] PDF(s)\n"
    )

    # ── Step 2: Chunk documents ─────────────────────────
    console.print("[bold]Step 2/4[/bold] Chunking pages...")
 
    chunker = Chunker(
        chunk_size    = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
 
    if not chunks:
        console.print("[bold red]No chunks created. Exiting.[/bold red]")
        return {}
 
    console.print(
        f"         Created [green]{len(chunks)}[/green] chunk(s)\n"
    )


    # ── Step 3: Embed chunks ────────────────────────────
    console.print("[bold]Step 3/4[/bold] Embedding chunks...")
    console.print(
        f"         Model: [cyan]{settings.embedding_model}[/cyan]"
    )
    console.print(
        "         First run downloads model (~500MB). "
        "Subsequent runs use cache.\n"
    )
 
    embedder = Embedder(
        model_name = settings.embedding_model,
        batch_size = settings.embedding_batch_size,
    )
    embedded_chunks = embedder.embed_chunks(chunks)
 
    console.print(
        f"         Embedded [green]{len(embedded_chunks)}[/green] "
        f"chunk(s)\n"
    )
 

    # ── Step 4: Store in Supabase ───────────────────────
    console.print("[bold]Step 4/4[/bold] Storing in Supabase...")
 
    store        = VectorStore()
    stored_count = store.add_chunks(
        embedded_chunks,
        skip_existing=True,
    )
 
    console.print(
        f"         Stored [green]{stored_count}[/green] new chunk(s) "
        f"([dim]{len(embedded_chunks) - stored_count} skipped "
        f"— already indexed[/dim])\n"
    )


    # ── Summary ─────────────────────────────────────────
    total_time = round(time.time() - start_time, 2)
 
    stats = {
        "documents_loaded"   : len(documents),
        "unique_files"       : _count_unique_files(documents),
        "chunks_created"     : len(chunks),
        "chunks_stored"      : stored_count,
        "chunks_skipped"     : len(embedded_chunks) - stored_count,
        "total_time_seconds" : total_time,
    }
 
    # print summary table
    _print_summary(stats, store)
 
    return stats

def _count_unique_files(documents: list) -> int:
    """Count unique source files in document list."""
    return len(set(doc.source for doc in documents))
 
def _print_summary(stats: dict, store: VectorStore) -> None:
    """Print a rich summary table after ingestion."""
 
    console.print("[bold green]Ingestion Complete![/bold green]\n")
 
    # stats table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="dim")
    table.add_column("Value",   justify="right")
 
    table.add_row("PDF files loaded",    str(stats["unique_files"]))
    table.add_row("Pages loaded",        str(stats["documents_loaded"]))
    table.add_row("Chunks created",      str(stats["chunks_created"]))
    table.add_row("Chunks stored",       str(stats["chunks_stored"]))
    table.add_row("Chunks skipped",      str(stats["chunks_skipped"]))
    table.add_row("Total time",          f"{stats['total_time_seconds']}s")
 
    console.print(table)
 
    # vector store stats
    try:
        db_stats = store.get_stats()
        console.print(
            f"\n[dim]Supabase total: "
            f"{db_stats['total_chunks']} chunks across "
            f"{db_stats['unique_sources']} documents[/dim]"
        )
 
        # show category breakdown
        if db_stats.get("categories"):
            console.print("\n[dim]Category breakdown:[/dim]")
            for cat, count in db_stats["categories"].items():
                console.print(f"  [dim]{cat}: {count} chunks[/dim]")
 
    except Exception as e:
        logger.warning(f"Could not fetch DB stats: {e}")
 

def main():


    # ── Setup MLflow ────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    
    # everything inside this block is tracked
    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_params({
            "chunk_size"      : settings.chunk_size,
            "chunk_overlap"   : settings.chunk_overlap,
            "embedding_model" : settings.embedding_model,
            "batch_size"      : settings.embedding_batch_size,
            "documents_path"  : "data/documents/procedures",
            "environment"     : settings.environment.value,
        })
        # run the pipeline
        stats = run_ingestion("data/documents/procedures")

        if not stats:
            console.print("[bold red]Ingestion failed.[/bold red]")
            sys.exit(1)

        # log results to MLflow
        mlflow.log_metrics({
            "documents_loaded"   : stats["documents_loaded"],
            "chunks_created"     : stats["chunks_created"],
            "chunks_stored"      : stats["chunks_stored"],
            "total_time_seconds" : stats["total_time_seconds"],
        })

        console.print(
            f"[dim]MLflow run logged to: "
            f"{settings.mlflow_tracking_uri}[/dim]\n"
        )


if __name__ == "__main__":
    main()
