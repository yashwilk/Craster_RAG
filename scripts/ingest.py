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
from rich.progress import Progress, SpinnerColumn, TextColumn

# add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from craster_rag.ingestion.txt_loader import TxtLoader
from craster_rag.ingestion.chunker import Chunker
from craster_rag.ingestion.embedder import Embedder
from craster_rag.retrieval.vector_store import VectorStore

logging.basicConfig(
    level  = getattr(logging, settings.log_level),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger  = logging.getLogger(__name__)
console = Console()


def run_ingestion(document_path: str = "data/documents/procedures") -> dict:
    start_time = time.time()
    console.print("\n[bold green]Craster RAG — Ingestion Pipeline[/bold green]")
    console.print(f"Documents path: {document_path}\n")

    # ── Step 1: Load documents ───────────────────────
    console.print("[bold]Step 1/4[/bold] Loading documents...")
    loader = TxtLoader()
    documents = loader.load(document_path)
    # Returns list[Document]
    if not documents:
        console.print("[bold red]No documents found. Exiting.[/bold red]")
        return {}
    console.print(
        f"         Loaded [green]{len(documents)}[/green] document(s)\n"
    )

    # ── Step 2: Chunk documents ─────────────────────────
    console.print("[bold]Step 2/4[/bold] Chunking documents...")
    chunker = Chunker(
        chunk_size    = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    # Returns list[Chunk]
    console.print(
        f"         Created [green]{len(chunks)}[/green] chunk(s)\n"
    )

    # ── Step 3: Embed chunks ────────────────────────────
    console.print("[bold]Step 3/4[/bold] Embedding chunks...")
    console.print(
        f"         Model: {settings.embedding_model}"
    )
    console.print(
        f"         This may take a few minutes on first run\n"
    )
    embedder = Embedder(
        model_name = settings.embedding_model,
        batch_size = settings.embedding_batch_size,
    )
    embedded_chunks = embedder.embed_chunks(chunks)
    # Returns list[EmbeddedChunk]

    console.print(
        f"         Embedded [green]{len(embedded_chunks)}[/green] chunk(s)\n"
    )

    # ── Step 4: Store in Supabase ───────────────────────
    console.print("[bold]Step 4/4[/bold] Storing in Supabase...")

    store        = VectorStore()
    stored_count = store.add_chunks(
        embedded_chunks,
        skip_existing=True,
    )

    console.print(
        f"         Stored [green]{stored_count}[/green] new chunk(s)\n"
    )

    total_time = round(time.time() - start_time, 2)

    stats = {
        "documents_loaded"   : len(documents),
        "chunks_created"     : len(chunks),
        "chunks_stored"      : stored_count,
        "total_time_seconds" : total_time,
    }

    console.print("[bold green]Ingestion complete![/bold green]")
    console.print(f"   Documents loaded : {stats['documents_loaded']}")
    console.print(f"   Chunks created   : {stats['chunks_created']}")
    console.print(f"   Chunks stored    : {stats['chunks_stored']}")
    console.print(f"   Total time       : {stats['total_time_seconds']}s\n")

    return stats


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
