from backend.vectorstore.chroma_store import ChromaStore


def test_chroma_store():
    store = ChromaStore(
        persist_directory="data/test_chroma",
        collection_name="test_collection"
    )

    chunks = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Federated learning protects data privacy."
        },
        {
            "chunk_id": 2,
            "page_number": 2,
            "text": "Intrusion detection identifies malicious network activity."
        }
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    store.add_chunks(chunks, embeddings)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1
    )

    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == (
        "Federated learning protects data privacy."
    )

    print("Retrieved:", results["documents"][0][0])
    print("ChromaDB test passed!")


if __name__ == "__main__":
    test_chroma_store()
