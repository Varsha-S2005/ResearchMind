from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


def test_retrieval():
    embedder = Embedder()
    store = ChromaStore()

    query = "What are the challenges of federated learning?"

    query_embedding = embedder.embed_text(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\nQuery:", query)
    print("\nRetrieved chunks:\n")

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        print(f"--- Result {i} ---")
        print(f"Page: {metadata['page_number']}")
        print(f"Distance: {distance}")
        print(f"Text: {document[:500]}")
        print()


if __name__ == "__main__":
    test_retrieval()
