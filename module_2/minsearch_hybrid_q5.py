from minsearch import Index, VectorSearch
from embedder import Embedder
from load_data import chunk

QUERY = "How do I give the model access to tools?"
TOP_K = 5


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


# --- Text index ---
text_index = Index(text_fields=["content"])
text_index.fit(chunk)

# --- Vector index ---
emb = Embedder()
texts = [c["content"] for c in chunk]
vectors = emb.encode_batch(texts)

v_index = VectorSearch()
v_index.fit(vectors, chunk)

# --- Run both searches ---
text_results = text_index.search(QUERY, num_results=TOP_K)
query_vec = emb.encode(QUERY)
vector_results = v_index.search(query_vec, num_results=TOP_K)

print(f"query: {QUERY}\n")

print(f"text search top {TOP_K}:")
for r in text_results:
    print(f"  {r['filename']}  start={r['start']}")

print(f"\nvector search top {TOP_K}:")
for r in vector_results:
    print(f"  {r['filename']}  start={r['start']}")

# --- Fuse with RRF ---
results = rrf([vector_results, text_results], k=60, num_results=TOP_K)

print(f"\nRRF top {TOP_K}:")
for i, r in enumerate(results):
    print(f"  {i + 1}. {r['filename']}  start={r['start']}")

print(f"\nrank #1 after RRF: {results[0]['filename']}")