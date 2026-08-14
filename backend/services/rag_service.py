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
    Manages the ResearchMind RAG pipeline.
    """

    def __init__(self):

        self.embedder = Embedder()

        self.vector_store = ChromaStore(
            persist_directory="data/chroma"
        )

        self.reranker = CrossEncoderReranker()

        self.llm = GeminiLLM()

        self.chunks = []

        self._load_initial_document()

        self._build_pipeline()

    def _load_initial_document(self):

        pdf_path = "data/papers/sample.pdf"

        pages = extract_text_from_pdf(pdf_path)

        self.chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=50
        )

    def _build_pipeline(self):

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        self.retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm
        )

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        return self.pipeline.answer(
            question,
            top_k=top_k
        )

    def ingest_pdf(
        self,
        pdf_path: str
    ) -> int:

        pages = extract_text_from_pdf(pdf_path)

        chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=50
        )

        embeddings = [
            self.embedder.embed_text(
                chunk["text"]
            )
            for chunk in chunks
        ]

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        self.chunks.extend(chunks)

        self._build_pipeline()

        return len(chunks)
