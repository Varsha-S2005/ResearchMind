from backend.generation.context_builder import ContextBuilder


def test_context_builder():

    results = [
        {
            "chunk": {
                "chunk_id": 1,
                "page_number": 2,
                "text": "Federated learning allows multiple clients to collaboratively train a model."
            },
            "score": 0.91
        },
        {
            "chunk": {
                "chunk_id": 7,
                "page_number": 6,
                "text": "Data heterogeneity is an important challenge in federated learning."
            },
            "score": 0.82
        }
    ]

    builder = ContextBuilder()

    context = builder.build(results)

    print("\nGenerated Context:\n")
    print(context)

    assert "Page: 2" in context
    assert "Page: 6" in context
    assert "Chunk ID: 1" in context
    assert "Chunk ID: 7" in context
    assert "Data heterogeneity" in context

    print("\nContext builder test passed!")


if __name__ == "__main__":
    test_context_builder()
