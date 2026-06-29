from minsearch import Index, VectorSearch
from embedder import Embedder
from load_data import chunk

QUERY = "How do I store vectors in PostgreSQL?"
TOP_K = 5

# --- Text index (BM25/TF-IDF style) ---
text_index = Index(text_fields=["content"])
text_index.fit(chunk)

# --- Vector index (semantic) ---
emb = Embedder()
texts = [c["content"] for c in chunk]
vectors = emb.encode_batch(texts)

v_index = VectorSearch()
v_index.fit(vectors, chunk)

# --- Run both searches ---
text_results = text_index.search(QUERY, num_results=TOP_K)
query_vec = emb.encode(QUERY)
vector_results = v_index.search(query_vec, num_results=TOP_K)

text_files = [r["filename"] for r in text_results]
vector_files = [r["filename"] for r in vector_results]

print(f"query: {QUERY}\n")
print(f"text search top {TOP_K} filenames:")
for f in text_files:
    print(f"  {f}")

print(f"\nvector search top {TOP_K} filenames:")
for f in vector_files:
    print(f"  {f}")

only_in_vector = [f for f in vector_files if f not in text_files]
print(f"\nin vector results but NOT in text results ({len(only_in_vector)}):")
for f in only_in_vector:
    print(f"  {f}")