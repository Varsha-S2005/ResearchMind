def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping
    word-based chunks.

    Each chunk contains:
    - document_id
    - chunk_id
    - page_number
    - text
    """

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    chunk_id = 1

    for page in pages:

        document_id = page["document_id"]
        page_number = page["page_number"]
        text = page["text"]

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            chunks.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page_number": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            if end == len(words):
                break

            start = end - overlap

    return chunks
