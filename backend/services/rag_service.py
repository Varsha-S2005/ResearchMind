from pathlib import Path

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

        # -------------------------------------------------
        # Initialize models and vector store
        # -------------------------------------------------

        self.embedder = Embedder()

        self.vector_store = ChromaStore(
            persist_directory="data/chroma"
        )

        self.reranker = CrossEncoderReranker()

        self.llm = GeminiLLM()

        # -------------------------------------------------
        # Load existing chunks from ChromaDB
        # -------------------------------------------------

        self.chunks = self._load_existing_chunks()

        # -------------------------------------------------
        # Build retrieval + RAG pipeline
        # -------------------------------------------------

        self._build_pipeline()

    # =====================================================
    # LOAD EXISTING CHUNKS
    # =====================================================

    def _load_existing_chunks(self) -> list[dict]:
        """
        Load existing document chunks from ChromaDB.

        This allows the application to restart without
        re-processing all PDFs.
        """

        if self.vector_store.collection.count() == 0:
            return []

        results = self.vector_store.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        chunks = []

        for text, metadata in zip(
            documents,
            metadatas
        ):

            chunks.append({
                "document_id": metadata["document_id"],
                "page_number": metadata["page_number"],
                "chunk_id": metadata["chunk_id"],
                "text": text
            })

        return chunks

    # =====================================================
    # BUILD PIPELINE
    # =====================================================

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

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        """
        Ask a research question using the RAG pipeline.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not self.chunks:
            return {
                "question": question,
                "answer": (
                    "No research documents are currently "
                    "available. Please upload a PDF first."
                ),
                "sources": []
            }

        return self.pipeline.answer(
            question,
            top_k=top_k
        )

    # =====================================================
    # INGEST PDF
    # =====================================================

    def ingest_pdf(
        self,
        pdf_path: str,
        filename: str
    ) -> int:
        """
        Ingest a new PDF into the RAG system.
        """

        document_id = Path(filename).stem

        # -------------------------------------------------
        # Extract PDF text
        # -------------------------------------------------

        pages = extract_text_from_pdf(
            pdf_path,
            document_id=document_id
        )

        if not pages:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        # -------------------------------------------------
        # Create chunks
        # -------------------------------------------------

        chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=50
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated from the PDF."
            )

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        embeddings = [
            self.embedder.embed_text(
                chunk["text"]
            )
            for chunk in chunks
        ]

        # -------------------------------------------------
        # Store in ChromaDB
        # -------------------------------------------------

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        # -------------------------------------------------
        # Update in-memory BM25 corpus
        # -------------------------------------------------

        self.chunks.extend(chunks)

        # -------------------------------------------------
        # Rebuild retrieval pipeline
        # -------------------------------------------------

        self._build_pipeline()

        return len(chunks)

    # =====================================================
    # LIST DOCUMENTS
    # =====================================================

    def list_documents(self) -> list[dict]:
        """
        Return all documents currently stored in ChromaDB.
        """

        return self.vector_store.list_documents()
