from minsearch import VectorSearch
from embedder import Embedder
from load_data import chunk

QUERY = "What metric do we use to evaluate a search engine?"

emb = Embedder()

texts = [c["content"] for c in chunk]
vectors = emb.encode_batch(texts)

v_index = VectorSearch()
v_index.fit(vectors, chunk)

query_vec = emb.encode(QUERY)
results = v_index.search(query_vec, num_results=1)

top = results[0]
print(f"query: {QUERY}")
print(f"top chunk filename: {top['filename']}")
print(f"top chunk start: {top['start']}")