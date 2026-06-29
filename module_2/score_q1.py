import numpy as np
from embedder import Embedder
from load_data import chunk
from tqdm.auto import tqdm
import numpy as np


QUERY = "How does approximate nearest neighbor search work?"

emb = Embedder()

# Embed every chunk's content in one batched call
# texts = [c["content"] for c in chunk]
# X = emb.encode_batch(texts)                 # shape: (n_chunks, 384)
# print(f"X shape: {X.shape}")

# # Embed the query
v = emb.encode(QUERY)                       # shape: (384,)
print(f"v shape: {v.shape}")



# # Cosine similarity = dot product (vectors are L2-normalized)
# scores = X.dot(v)                           # shape: (n_chunks,)

# print("scores (top 5):")
# for i in np.argsort(scores)[::-1][:5]:
#     snippet = chunk[i]["content"][:80].replace("\n", " ")
#     print(f"  {scores[i]:.4f}  [{i}] {snippet}...")

# print("\nall scores:")
# print(scores)



batch_size = 50
vectors = []
for i in tqdm(range(0, len(chunk), batch_size)):
    batch = chunk[i:i + batch_size]
    texts = [c["content"] for c in batch]
    batch_vectors = emb.encode_batch(texts)
    vectors.append(batch_vectors)
X = np.concatenate(vectors, axis=0)
scores = X.dot(v)
print(scores)

# Top chunk -> which filename?
top_idx = int(np.argmax(scores))
top_chunk = chunk[top_idx]
print(f"\ntop score: {scores[top_idx]:.4f}")
print(f"top chunk idx: {top_idx}")
print(f"top chunk filename: {top_chunk['filename']}")
print(f"top chunk start: {top_chunk['start']}")