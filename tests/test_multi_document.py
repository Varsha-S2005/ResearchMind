from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


def test_multi_document():

    # --------------------------------
    # 1. Load PDF
    # --------------------------------
    pdf_path = "data/papers/sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    # --------------------------------
    # 2. Add document metadata
    # --------------------------------
    document_id = "test_document_001"

    for chunk in chunks:
        chunk["document_id"] = document_id
        chunk["filename"] = "sample.pdf"

    print(f"Generated {len(chunks)} chunks.")

    # --------------------------------
    # 3. Create embeddings
    # --------------------------------
    embedder = Embedder()

    embeddings = [
        embedder.embed_text(chunk["text"])
        for chunk in chunks
    ]

    # --------------------------------
    # 4. Store in ChromaDB
    # --------------------------------
    vector_store = ChromaStore(
        persist_directory="data/test_chroma",
        collection_name="multi_document_test"
    )

    vector_store.add_chunks(
        chunks,
        embeddings
    )

    # --------------------------------
    # 5. Search
    # --------------------------------
    query = "What are the challenges of federated learning?"

    query_embedding = embedder.embed_text(query)

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    # --------------------------------
    # 6. Validate results
    # --------------------------------
    assert len(results["documents"][0]) > 0

    for metadata in results["metadatas"][0]:

        assert "document_id" in metadata
        assert "filename" in metadata
        assert "page_number" in metadata
        assert "chunk_id" in metadata

        print(
            f"Document: {metadata['filename']} | "
            f"Chunk: {metadata['chunk_id']}"
        )

    print("\nMulti-document metadata test passed!")

if __name__ == "__main__":
    test_multi_document()

