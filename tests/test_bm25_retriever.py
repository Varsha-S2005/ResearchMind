from backend.retrieval.bm25_retriever import BM25Retriever


def test_bm25_retriever():
    chunks = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Federated learning improves privacy."
        },
        {
            "chunk_id": 2,
            "page_number": 2,
            "text": "Intrusion detection identifies malicious attacks."
        },
        {
            "chunk_id": 3,
            "page_number": 3,
            "text": "Vehicular networks face intrusion attacks."
        }
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "vehicular intrusion attacks",
        top_k=2
    )

    print("\nBM25 Results:\n")

    for result in results:
        print(
            f"Chunk {result['chunk']['chunk_id']} "
            f"| Score: {result['score']:.4f}"
        )
        print(result["chunk"]["text"])
        print()

    assert len(results) == 2
    assert results[0]["chunk"]["chunk_id"] == 3

    print("BM25 test passed!")


if __name__ == "__main__":
    test_bm25_retriever()
