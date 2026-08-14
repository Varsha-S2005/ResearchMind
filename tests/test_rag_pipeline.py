from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from backend.reranking.cross_encoder import CrossEncoderReranker

from backend.generation.gemini_llm import GeminiLLM
from backend.rag.rag_pipeline import RAGPipeline


def test_rag_pipeline():

    # --------------------------------
    # 1. Load PDF
    # --------------------------------
    pdf_path = "data/papers/sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages.")

    # --------------------------------
    # 2. Create chunks
    # --------------------------------
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=50
    )

    print(f"Generated {len(chunks)} chunks.")

    # --------------------------------
    # 3. Initialize embedding model
    # --------------------------------
    embedder = Embedder()

    # --------------------------------
    # 4. Initialize ChromaDB
    # --------------------------------
    vector_store = ChromaStore(
        persist_directory="data/chroma"
    )

    # --------------------------------
    # 5. Initialize BM25
    # --------------------------------
    bm25_retriever = BM25Retriever(
        chunks
    )

    # --------------------------------
    # 6. Initialize Hybrid Retriever
    # --------------------------------
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder
    )

    # --------------------------------
    # 7. Initialize Cross Encoder
    # --------------------------------
    reranker = CrossEncoderReranker()

    # --------------------------------
    # 8. Initialize Gemini
    # --------------------------------
    llm = GeminiLLM()

    # --------------------------------
    # 9. Create RAG Pipeline
    # --------------------------------
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm=llm
    )

    # --------------------------------
    # 10. Ask question
    # --------------------------------
    question = (
        "What are the challenges of federated learning "
        "in vehicular networks?"
    )

    result = pipeline.answer(
        question,
        top_k=5
    )

    # --------------------------------
    # 11. Display answer
    # --------------------------------
    print("\n" + "=" * 70)
    print("RESEARCH QUESTION")
    print("=" * 70)

    print(question)

    print("\n" + "=" * 70)
    print("GEMINI ANSWER")
    print("=" * 70)

    print(result["answer"])

    # --------------------------------
    # 12. Display sources
    # --------------------------------
    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

	    print(
        	f"\nSource: {source['source_id']}"
        	f"\nPage: {source['page_number']}"
        	f"\nChunk ID: {source['chunk_id']}"
        	f"\nScore: {source['score']}"
    		)
    # --------------------------------
    # 13. Assertions
    # --------------------------------
    assert result["answer"]
    assert len(result["sources"]) > 0

    print("\nRAG pipeline test passed!")


if __name__ == "__main__":
    test_rag_pipeline()
