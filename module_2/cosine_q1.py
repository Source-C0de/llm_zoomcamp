import numpy as np
from embedder import Embedder

QUERY = "How does approximate nearest neighbor search work?"
DOC_PATH = "07-sqlitesearch-vector.md"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()

print(f"doc length (chars): {len(doc_text)}")

emb = Embedder()
q_vec = emb.encode(QUERY)
d_vec = emb.encode(doc_text)

print(f"query vector dim: {len(q_vec)}")
print(f"doc   vector dim: {len(d_vec)}")

cosine = float(np.dot(q_vec, d_vec))
print(f"cosine similarity: {cosine}")