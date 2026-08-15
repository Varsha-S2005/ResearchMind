from backend.services.rag_service import RAGService


def test_rag_pipeline():

    print("=" * 70)
    print("ResearchMind End-to-End RAG Test")
    print("=" * 70)

    # -------------------------------------------------
    # Initialize RAG service
    # -------------------------------------------------

    print("\nInitializing ResearchMind...")

    service = RAGService()

    print("RAG system initialized successfully.")

    # -------------------------------------------------
    # Question
    # -------------------------------------------------

    question = (
        "What are the main challenges of federated learning "
        "in vehicular networks?"
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # -------------------------------------------------
    # Run RAG pipeline
    # -------------------------------------------------

    result = service.ask(
        question,
        top_k=5
    )

    # -------------------------------------------------
    # Basic validation
    # -------------------------------------------------

    assert result is not None

    assert "question" in result
    assert "answer" in result
    assert "sources" in result

    assert result["question"] == question

    assert isinstance(
        result["answer"],
        str
    )

    assert len(
        result["answer"].strip()
    ) > 0

    assert isinstance(
        result["sources"],
        list
    )

    assert len(
        result["sources"]
    ) > 0

    # -------------------------------------------------
    # Validate source structure
    # -------------------------------------------------

    for source in result["sources"]:

        assert "source_id" in source
        assert "document_id" in source
        assert "page_number" in source
        assert "chunk_id" in source
        assert "score" in source

    # -------------------------------------------------
    # Display generated answer
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(result["answer"])

    # -------------------------------------------------
    # Display sources
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"{source['source_id']} | "
            f"Document={source['document_id']} | "
            f"Page={source['page_number']} | "
            f"Chunk={source['chunk_id']} | "
            f"Score={source['score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("RAG TEST PASSED")
    print("=" * 70)
