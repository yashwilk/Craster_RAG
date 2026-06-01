"""

# INPUT — list of Chunk objects
chunks = [
    Chunk(content="Step 1: Go to User Management..."),
    Chunk(content="Step 2: Click Add New User..."),
    Chunk(content="Step 3: Fill in details..."),
]

# OUTPUT — same chunks with embeddings attached
chunks = [
    Chunk(content="Step 1...", embedding=[0.21, 0.84, 0.11, ...]),
    Chunk(content="Step 2...", embedding=[0.22, 0.83, 0.12, ...]),
    Chunk(content="Step 3...", embedding=[0.19, 0.81, 0.14, ...]),
]
chunk has 500 tokens    → still 768 numbers output
"""


import logging
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from craster_rag.ingestion.chuker import Chunk


logger = logging.getLogger(__name__)

@dataclass
class EmbeddedChunk:
    content: str
    source: str
    title: str
    doc_type: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = f"{self.source}::chunk_{self.chunk_index}"

    def __repr__(self) -> str:
        embed_status = (
            f"embedded({len(self.embedding)}d)"
            if self.embedding
            else "not embedded"
        )
        return (
            f"EmbeddedChunk("
            f"title='{self.title}', "
            f"chunk={self.chunk_index + 1}/{self.total_chunks}, "
            f"tokens={self.token_count}, "
            f"{embed_status})"
        )


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading embedding model '{model_name}' "
            f"on {self.device}..."
        )

        try:
            self._model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            logger.warning("No chunks provided to embedder")
            return []

        #texts-loop through each chunk ina extract the chunk.content as array.so each values in the array is a chunk.
        texts = [chunk.content for chunk in chunks]

        embeddings = self._embed_text(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(EmbeddedChunk(
                content=chunk.content,
                source=chunk.source,
                title=chunk.title,
                doc_type=chunk.doc_type,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                chunk_id=chunk.chunk_id,
                embedding=embedding,
            ))

        return embedded_chunks

    def _embed_text(self, texts: list[str]) -> list[list[float]]:
        #tokenize(conver text to numbers)->numbers pass throufg neurat network to understand relationship->Pooling all token vectors collapse to one vector per chunk-> normalize-scale between -1 and 1->return list of list of floats
        try:
            prefixed_texts = [
                f"Represent this passage: {text}"
                for text in texts
            ]

            embeddings = self._model.encode(
                prefixed_texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) >= 10)

            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_query(self, query: str) -> list[float]:
        """Used at query time when a user asks a question.
        The query embedding is compared against chunk
        embeddings to find the most relevant chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Embedding query: '{query[:50]}...'")
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self._model.encode(
            prefixed_query, normalize_embeddings=True, convert_to_numpy=True)
        return embedding.tolist()
