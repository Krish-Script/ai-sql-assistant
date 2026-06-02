"""
retriever.py -- Task 5: Cosine similarity retriever for RAG
Loads pre-computed embeddings and returns top-k most similar pairs.

Used by: api.py (/generate with use_rag=true)

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.retrieve("Get all users from France", k=3)
    for item in results:
        print(item["similarity"], item["nl_query"], item["sql_query"])

Key design decisions:
    - Embeddings are L2-normalized at creation time (embed_pairs.py)
      so cosine similarity = dot product (fast, no sqrt needed)
    - exclude_ids prevents returning the query's own pair when
      the query is already in the index (avoids data leakage)
    - Retriever loads once at module level in api.py, not per request
"""

import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------
# Config
# -------------------------------------------------------

EMBEDDINGS_PATH = Path("data/embeddings/pairs_embeddings.npy")
INDEX_PATH      = Path("data/embeddings/pairs_index.json")
MODEL_NAME      = "all-MiniLM-L6-v2"

DEFAULT_K       = 3
MIN_SIMILARITY  = 0.0    # no hard threshold -- always return k results


# -------------------------------------------------------
# RetrievalResult
# -------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single retrieved pair with its similarity score.

    Attributes:
        idx        : position in the index (0-based)
        nl_query   : natural language question
        sql_query  : corresponding SQL query
        scenario   : scenario tag (filters, joins, etc.)
        difficulty : easy / medium / hard
        source     : spider or wikisql
        similarity : cosine similarity score (0.0 - 1.0)
    """
    idx       : int
    nl_query  : str
    sql_query : str
    scenario  : str
    difficulty: str
    source    : str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "idx"       : self.idx,
            "nl_query"  : self.nl_query,
            "sql_query" : self.sql_query,
            "scenario"  : self.scenario,
            "difficulty": self.difficulty,
            "source"    : self.source,
            "similarity": round(float(self.similarity), 4),
        }


# -------------------------------------------------------
# Retriever
# -------------------------------------------------------

class Retriever:
    """
    Cosine similarity retriever over pre-computed NL query embeddings.

    Load once at startup — embedding model and index stay in memory.

    Args:
        embeddings_path : path to .npy embeddings file
        index_path      : path to pairs_index.json
        model_name      : sentence-transformers model name
    """

    def __init__(
        self,
        embeddings_path: str | Path = EMBEDDINGS_PATH,
        index_path     : str | Path = INDEX_PATH,
        model_name     : str        = MODEL_NAME,
    ):
        self._embeddings_path = Path(embeddings_path)
        self._index_path      = Path(index_path)
        self._model_name      = model_name

        self._embeddings : np.ndarray      = None
        self._index      : list[dict]      = None
        self._model      : SentenceTransformer = None
        self._loaded     : bool            = False

        self._load()

    def _load(self) -> None:
        """Load embeddings, index, and model from disk."""
        if not self._embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self._embeddings_path}. "
                f"Run embed_pairs.py first."
            )
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self._index_path}. "
                f"Run embed_pairs.py first."
            )

        self._embeddings = np.load(self._embeddings_path)
        with open(self._index_path, encoding="utf-8") as f:
            self._index = json.load(f)

        # Verify alignment
        assert len(self._embeddings) == len(self._index), (
            f"Embeddings ({len(self._embeddings)}) and index "
            f"({len(self._index)}) are out of sync. "
            f"Re-run embed_pairs.py."
        )

        self._model  = SentenceTransformer(self._model_name)
        self._loaded = True

    def retrieve(
        self,
        query      : str,
        k          : int       = DEFAULT_K,
        exclude_ids: list[int] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k most similar pairs for a given NL query.

        Args:
            query       : natural language question to find similar pairs for
            k           : number of results to return
            exclude_ids : list of index positions to exclude from results
                          (use this to prevent returning the query's own pair)

        Returns:
            List of RetrievalResult sorted by similarity descending.
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call _load() first.")

        exclude_set = set(exclude_ids or [])

        # Encode query with L2 normalization (matches stored embeddings)
        query_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity = dot product (both sides L2-normalized)
        scores = self._embeddings @ query_emb  # shape: (n_pairs,)

        # Sort descending, exclude specified ids
        sorted_idxs = np.argsort(scores)[::-1]
        results     = []

        for idx in sorted_idxs:
            if int(idx) in exclude_set:
                continue
            if len(results) >= k:
                break

            entry = self._index[idx]
            results.append(RetrievalResult(
                idx       = int(idx),
                nl_query  = entry["nl_query"],
                sql_query = entry["sql_query"],
                scenario  = entry["scenario"],
                difficulty= entry["difficulty"],
                source    = entry["source"],
                similarity= float(scores[idx]),
            ))

        return results

    def retrieve_dicts(
        self,
        query      : str,
        k          : int       = DEFAULT_K,
        exclude_ids: list[int] = None,
    ) -> list[dict]:
        """
        Convenience wrapper — returns list of dicts instead of dataclasses.
        Used by api.py for JSON serialization.
        """
        return [r.to_dict() for r in self.retrieve(query, k, exclude_ids)]

    @property
    def size(self) -> int:
        """Number of pairs in the index."""
        return len(self._index) if self._index else 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# -------------------------------------------------------
# Demo / sanity check
# -------------------------------------------------------

if __name__ == "__main__":
    print("Retriever sanity check")
    print("=" * 55)

    r = Retriever()
    print(f"Loaded {r.size} pairs\n")

    test_queries = [
        "Get all users from France",
        "How many orders does each customer have?",
        "Find products that have never been sold",
        "What is the average salary by department?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print(f"{'-'*55}")
        results = r.retrieve(query, k=3)
        for i, res in enumerate(results, 1):
            print(f"  [{i}] sim={res.similarity:.4f}  [{res.scenario}]"
                  f"  {res.nl_query[:55]}")
            print(f"      SQL: {res.sql_query[:60]}")
        print()

    # Test exclude_ids -- verify own pair is excluded
    print("exclude_ids test:")
    print(f"{'-'*55}")
    # Use first pair's NL as query -- should not appear in results
    first_nl  = r._index[0]["nl_query"]
    with_self    = r.retrieve(first_nl, k=3)
    without_self = r.retrieve(first_nl, k=3, exclude_ids=[0])

    print(f"Query: {first_nl[:55]}")
    print(f"With self    [0]: sim={with_self[0].similarity:.4f}  "
          f"{with_self[0].nl_query[:45]}")
    print(f"Without self [0]: sim={without_self[0].similarity:.4f}  "
          f"{without_self[0].nl_query[:45]}")

    self_excluded = all(r.idx != 0 for r in without_self)
    print(f"Self excluded: {self_excluded}")
    assert self_excluded, "exclude_ids not working!"

    print(f"\n{'='*55}")
    print("Retriever sanity check complete!")