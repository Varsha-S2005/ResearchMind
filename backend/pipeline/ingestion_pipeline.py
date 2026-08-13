from backend.ingestion.pdf_parser import extract_text_from_pdf
from backend.chunking.text_chunker import chunk_pages
from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


class IngestionPipeline:
    """
    Coordinates PDF parsing, chunking, embedding,
    and vector storage.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = ChromaStore()

    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF and store its chunks and embeddings.
        """

        # Step 1: Extract text from PDF
        pages = extract_text_from_pdf(pdf_path)

        # Step 2: Split pages into chunks
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No text chunks were generated from the PDF.")

        # Step 3: Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Store chunks and embeddings
        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings
        )

        print(f"Processed {len(pages)} pages.")
        print(f"Generated {len(chunks)} chunks.")
        print(f"Stored {len(embeddings)} embeddings.")
