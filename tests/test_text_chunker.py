from backend.chunking.text_chunker import chunk_pages


def test_chunk_pages():

    pages = [
        {
            "document_id": "test_document",
            "page_number": 1,
            "text": " ".join(
                f"word{i}"
                for i in range(1, 1201)
            )
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert chunk["document_id"] == "test_document"
        assert "chunk_id" in chunk
        assert "page_number" in chunk
        assert "text" in chunk
        assert len(chunk["text"].split()) > 0

    assert chunks[0]["chunk_id"] == 1
    assert chunks[0]["page_number"] == 1
