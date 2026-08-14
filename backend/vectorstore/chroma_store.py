import chromadb


class ChromaStore:
    """
    Handles storing and searching document embeddings using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "research_documents"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        ids = []

        documents = []

        metadatas = []

        for chunk in chunks:

            document_id = chunk.get(
                "document_id",
                "default_document"
            )

            chunk_id = chunk["chunk_id"]

            unique_id = (
                f"{document_id}_chunk_{chunk_id}"
            )

            ids.append(unique_id)

            documents.append(
                chunk["text"]
            )

            metadatas.append({
                "document_id": document_id,
                "filename": chunk.get(
                    "filename",
                    "unknown.pdf"
                ),
                "page_number": chunk["page_number"],
                "chunk_id": chunk_id
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> dict:

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
