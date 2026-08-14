from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from backend.reranking.cross_encoder import CrossEncoderReranker

from backend.generation.gemini_llm import GeminiLLM
from backend.rag.rag_pipeline import RAGPipeline


class RAGService:
    """
    Initializes and provides access to the ResearchMind RAG pipeline.
    """

    def __init__(self):
        # 1. Load research document
        pdf_path = "data/papers/sample.pdf"

        pages = extract_text_from_pdf(pdf_path)

        # 2. Create chunks
        chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=50
        )

        # 3. Initialize embedding model
        embedder = Embedder()

        # 4. Initialize vector store
        vector_store = ChromaStore(
            persist_directory="data/chroma"
        )

        # 5. Initialize BM25
        bm25_retriever = BM25Retriever(chunks)

        # 6. Initialize hybrid retriever
        retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_store=vector_store,
            embedder=embedder
        )

        # 7. Initialize reranker
        reranker = CrossEncoderReranker()

        # 8. Initialize Gemini
        llm = GeminiLLM()

        # 9. Create RAG pipeline
        self.pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            llm=llm
        )

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        """
        Ask a question using the ResearchMind RAG pipeline.
        """

        return self.pipeline.answer(
            question,
            top_k=top_k
        )
