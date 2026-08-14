from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.reranking.cross_encoder import CrossEncoderReranker


def test_cross_encoder():

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
    # 2. Create retrieval layers
    # -------------------------
    bm25 = BM25Retriever(chunks)

    embedder = Embedder()
    vector_store = ChromaStore()

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_store=vector_store,
        embedder=embedder
    )

    # -------------------------
    # 3. Retrieve candidates
    # -------------------------
    query = "What are the challenges of federated learning?"

    candidates = hybrid.search(
        query=query,
        top_k=5
    )

    print("\nBefore reranking:\n")

    for rank, result in enumerate(candidates, start=1):

        chunk = result["chunk"]

        print(f"--- Candidate {rank} ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Text: {chunk['text'][:300]}")
        print()

    # -------------------------
    # 4. Create reranker
    # -------------------------
    reranker = CrossEncoderReranker()

    # -------------------------
    # 5. Rerank candidates
    # -------------------------
    reranked = reranker.rerank(
        query=query,
        results=candidates,
        top_k=5
    )

    # -------------------------
    # 6. Display reranked results
    # -------------------------
    print("\nAfter reranking:\n")

    for rank, result in enumerate(reranked, start=1):

        chunk = result["chunk"]

        print(f"--- Result {rank} ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Cross-Encoder Score: {result['score']:.6f}")
        print(f"Text: {chunk['text'][:300]}")
        print()

    assert len(reranked) <= 5

    print("Cross-encoder reranking test passed!")


if __name__ == "__main__":
    test_cross_encoder()
