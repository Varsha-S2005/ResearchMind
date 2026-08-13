from backend.chunking.text_chunker import chunk_pages


def test_chunk_pages():
    pages = [
        {
            "page_number": 1,
            "text": " ".join(f"word{i}" for i in range(1, 1201))
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    assert len(chunks) == 3

    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == 1

    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 500
    assert len(chunks[2]["text"].split()) == 300

    # Check that the 50-word overlap exists
    first_chunk_words = chunks[0]["text"].split()
    second_chunk_words = chunks[1]["text"].split()

    assert first_chunk_words[-50:] == second_chunk_words[:50]


if __name__ == "__main__":
    test_chunk_pages()
    print("Text chunker test passed!")
