from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


def prepare_document(pdf_path, document_id, filename):
    pages = extract_text_from_pdf(pdf_path)

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    for chunk in chunks:
        chunk["document_id"] = document_id
        chunk["filename"] = filename

    return chunks


def test_two_documents():

    document_1 = prepare_document(
        "data/papers/sample.pdf",
        "document_001",
        "sample.pdf"
    )

    document_2 = prepare_document(
        "data/papers/paper2.pdf",
        "document_002",
        "paper2.pdf"
    )

    all_chunks = document_1 + document_2

    print("Document 1 chunks:", len(document_1))
    print("Document 2 chunks:", len(document_2))

    embedder = Embedder()

    embeddings = []

    for chunk in all_chunks:
        embedding = embedder.embed_text(chunk["text"])
        embeddings.append(embedding)

    vector_store = ChromaStore(
        persist_directory="data/test_chroma",
        collection_name="two_document_test"
    )

    vector_store.add_chunks(
        all_chunks,
        embeddings
    )

    query = "What are the challenges discussed in the research?"

    query_embedding = embedder.embed_text(query)

    results = vector_store.search(
        query_embedding,
        top_k=10
    )

    assert len(results["documents"][0]) > 0

    document_ids = set()

    for metadata in results["metadatas"][0]:

        assert "document_id" in metadata
        assert "filename" in metadata
        assert "page_number" in metadata
        assert "chunk_id" in metadata

        document_ids.add(metadata["document_id"])

        print(
            "Document:",
            metadata["filename"],
            "| ID:",
            metadata["document_id"],
            "| Page:",
            metadata["page_number"],
            "| Chunk:",
            metadata["chunk_id"]
        )

    print()
    print("Documents found:", document_ids)

    print()
    print("Two-document test passed!")


if __name__ == "__main__":
    test_two_documents()
