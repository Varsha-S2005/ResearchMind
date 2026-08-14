from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retrieval():

    # -------------------------
    # 1. Load and chunk PDF
    # -------------------------
    pages = extract_text_from_pdf(
        "data/papers/sample.pdf"
    )

    chunks = chunk_pages(pages)

    print(f"Loaded {len(pages)} pages.")
    print(f"Generated {len(chunks)} chunks.")

    # -------------------------
    # 2. Create BM25 retriever
    # -------------------------
    bm25 = BM25Retriever(chunks)

    # -------------------------
    # 3. Create embedding components
    # -------------------------
    embedder = Embedder()
    vector_store = ChromaStore()

    # -------------------------
    # 4. Create hybrid retriever
    # -------------------------
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_store=vector_store,
        embedder=embedder
    )

    # -------------------------
    # 5. Search
    # -------------------------
    query = "What are the challenges of federated learning?"

    results = hybrid.search(
        query=query,
        top_k=5
    )

    # -------------------------
    # 6. Display results
    # -------------------------
    print("\nQuery:")
    print(query)

    print("\nHybrid Retrieval Results:\n")

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print(f"--- Result {rank} ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Text: {chunk['text'][:500]}")
        print()

    assert len(results) <= 5

    print("Hybrid retrieval test passed!")


if __name__ == "__main__":
    test_hybrid_retrieval()
