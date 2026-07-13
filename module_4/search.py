"""Hybrid search for the llm-zoomcamp module-4 evaluation homework.

Rebuilds the homework-2 hybrid search (text + vector + RRF) over the
module-4 lesson chunks.

Vector search uses a TF-IDF + cosine similarity backend (no external
embedding model required). Drop in any embedder that exposes
`encode_batch(texts) -> np.ndarray` to swap to a semantic model.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

from embedder import Embedder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_OWNER = "DataTalksClub"
REPO_NAME = "llm-zoomcamp"
COMMIT_ID = "8c1834d"

TEXT_FIELDS = ["content"]
KEYWORD_FIELDS = ["filename"]

CACHE_DIR = Path(__file__).parent
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
VECTOR_INDEX_CACHE = CACHE_DIR / "vector_index.pkl"


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def fetch_chunks() -> list[dict]:
    """Fetch and chunk the module-4 lesson markdown files via gitsource."""
    reader = GithubRepositoryDataReader(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        commit_id=COMMIT_ID,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    return chunk_documents(documents, size=2000, step=1000)


def load_chunks(force: bool = False) -> list[dict]:
    """Load chunks from disk cache, or build and cache them on first run."""
    if force or not CHUNKS_CACHE.exists():
        chunks = fetch_chunks()
        with CHUNKS_CACHE.open("wb") as f:
            pickle.dump(chunks, f)
        return chunks

    with CHUNKS_CACHE.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------


def build_text_index(chunks: list[dict]) -> Index:
    """Build an in-memory BM25-style text index."""
    return Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS).fit(chunks)


def build_vector_index(
    chunks: list[dict], embedder: Embedder, force: bool = False
) -> VectorSearch:
    """Build a VectorSearch index over chunk embeddings.

    The fitted index (with embedded vectors) is cached to disk to avoid
    re-paying the embedding cost on subsequent runs.
    """
    if not force and VECTOR_INDEX_CACHE.exists():
        with VECTOR_INDEX_CACHE.open("rb") as f:
            return pickle.load(f)

    vectors = embedder.encode_batch([c["content"] for c in chunks], normalize=True)

    index = VectorSearch(keyword_fields=KEYWORD_FIELDS)
    index.fit(vectors, chunks)

    with VECTOR_INDEX_CACHE.open("wb") as f:
        pickle.dump(index, f)

    return index


# ---------------------------------------------------------------------------
# Singletons (lazy)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    return load_chunks()


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def _text_index() -> Index:
    return build_text_index(_chunks())


@lru_cache(maxsize=1)
def _vector_index() -> VectorSearch:
    return build_vector_index(_chunks(), _embedder())


# ---------------------------------------------------------------------------
# Search wrappers
# ---------------------------------------------------------------------------


def text_search(query: str, num_results: int = 5) -> list[dict]:
    """BM25-style text search over chunks."""
    return _text_index().search(query, num_results=num_results)


def vector_search(query: str, num_results: int = 5) -> list[dict]:
    """Vector (TF-IDF cosine) search over chunk embeddings."""
    query_vec = _embedder().encode(query, normalize=True)
    return _vector_index().search(query_vec, num_results=num_results)


def rrf(result_lists: list[list[dict]], k: int = 60, num_results: int = 5) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Chunks are keyed on `(filename, start)` so identical chunks coming from
    different retrievers are merged rather than duplicated.
    """
    scores: dict[tuple[str, int], float] = {}
    docs: dict[tuple[str, int], dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(
    query: str, k: int = 60, num_results: int = 5
) -> list[dict]:
    """Run text + vector retrieval and fuse with RRF."""
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k, num_results=num_results)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    from pprint import pprint

    query = "What exactly is a retrieval-augmented generation system?"

    print("=" * 70)
    print(f"QUERY: {query}\n")

    print("TEXT SEARCH (top 5):")
    for doc in text_search(query, num_results=5):
        print(f"  - {doc['filename']} @ start={doc['start']}")

    print("\nVECTOR SEARCH (top 5):")
    for doc in vector_search(query, num_results=5):
        print(f"  - {doc['filename']} @ start={doc['start']}")

    print("\nHYBRID SEARCH (top 5):")
    results = hybrid_search(query)
    for doc in results:
        print(f"  - {doc['filename']} @ start={doc['start']}")

    print()
    pprint(results[0])


if __name__ == "__main__":
    _smoke_test()