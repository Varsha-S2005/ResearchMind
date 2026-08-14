import chromadb


class ChromaStore:
    """
    Handles storing and searching document embeddings
    using ChromaDB.
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
        Store chunks, embeddings and metadata in ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        ids = [
            f"{chunk['document_id']}_chunk_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": chunk["document_id"],
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

        if self.collection.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        top_k = min(
            top_k,
            self.collection.count()
        )

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def list_documents(self) -> list[dict]:
        """
        Return a list of unique documents stored in ChromaDB.
        """

        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            include=["metadatas"]
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        documents = {}

        for metadata in metadatas:

            document_id = metadata.get(
                "document_id"
            )

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "chunk_count": 0
                }

            documents[document_id]["chunk_count"] += 1

        return list(documents.values())
