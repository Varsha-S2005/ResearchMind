from backend.embeddings.embedder import Embedder


def test_embedder():
    embedder = Embedder()

    text = "Federated learning improves privacy in vehicular networks."

    embedding = embedder.embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384

    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
    print("Embedding test passed!")


if __name__ == "__main__":
    test_embedder()
