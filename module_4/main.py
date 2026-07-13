"""Entrypoint for module-4 hybrid search demo."""

from search import hybrid_search, text_search, vector_search


def main() -> None:
    query = "What exactly is a retrieval-augmented generation system, and why does it help with answers that the model wouldn't know on its own?"

    print(f"Query: {query}\n")
    print("Top 5 hybrid results:")
    for rank, doc in enumerate(hybrid_search(query), start=1):
        print(f"  {rank}. {doc['filename']} (start={doc['start']})")


if __name__ == "__main__":
    main()