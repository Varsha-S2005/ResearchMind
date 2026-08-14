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
        """
        Store chunks, embeddings, and metadata in ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        ids = [
            f"chunk_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

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
        """
        Search for the most similar chunks.
        """

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
