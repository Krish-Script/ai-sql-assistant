"""
embed_pairs.py -- Task 4: Embed all 64 NL queries using sentence-transformers
Reads  : data/processed/pairs_clean.json
Writes : data/embeddings/pairs_embeddings.npy  -- numpy array of shape (64, 384)
         data/embeddings/pairs_index.json       -- metadata index for each embedding

Model: all-MiniLM-L6-v2 (384-dim, fast, good semantic similarity)
       Downloads ~80MB on first run, cached locally after that.

Run:
    python src/embed_pairs.py
"""

import json
import sys
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Config
# -------------------------------------------------------

PAIRS_PATH      = Path("data/processed/pairs_clean.json")
EMBEDDINGS_DIR  = Path("data/embeddings")
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "pairs_embeddings.npy"
INDEX_PATH      = EMBEDDINGS_DIR / "pairs_index.json"

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~80MB, fast


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    # Load pairs
    if not PAIRS_PATH.exists():
        print(f"ERROR: {PAIRS_PATH} not found. Run clean.py first.")
        sys.exit(1)

    with open(PAIRS_PATH, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs from {PAIRS_PATH}")

    # Load model
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads ~80MB — cached after that)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Extract NL queries to embed
    # We embed NL queries (not SQL) because retrieval is query-driven
    nl_queries = [p["nl_query"] for p in pairs]

    # Generate embeddings
    print(f"\nEmbedding {len(nl_queries)} NL queries...")
    embeddings = model.encode(
        nl_queries,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Build index -- metadata for each embedding row
    index = []
    for i, pair in enumerate(pairs):
        index.append({
            "idx"       : i,
            "nl_query"  : pair["nl_query"],
            "sql_query" : pair["sql_query"],
            "scenario"  : pair["scenario"],
            "difficulty": pair["difficulty"],
            "source"    : pair["source"],
        })

    # Save
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nEmbeddings saved -> {EMBEDDINGS_PATH.resolve()}")
    print(f"  Shape : {embeddings.shape}")
    print(f"  dtype : {embeddings.dtype}")
    print(f"  Size  : {EMBEDDINGS_PATH.stat().st_size / 1024:.1f} KB")

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Index   saved -> {INDEX_PATH.resolve()}")
    print(f"  Entries: {len(index)}")

    # Sanity check -- show top-3 most similar to first query
    print(f"\nSanity check -- top 3 similar to: '{nl_queries[0][:60]}'")
    query_emb  = embeddings[0]
    scores     = embeddings @ query_emb   # dot product = cosine sim (normalized)
    top_idxs   = np.argsort(scores)[::-1][:4]  # top 4 (first is self)

    for rank, idx in enumerate(top_idxs):
        marker = "(self)" if idx == 0 else ""
        print(f"  [{rank+1}] score={scores[idx]:.4f} {marker}  {nl_queries[idx][:60]}")

    print(f"\nTask 4 complete!")


if __name__ == "__main__":
    main()