class HybridRetriever:
    """
    Combines BM25 and dense retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_retriever,
        vector_store,
        embedder
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
        rrf_k: int = 60
    ) -> list[dict]:
        """
        Perform hybrid retrieval using:

        1. BM25 lexical retrieval
        2. Dense vector retrieval
        3. Reciprocal Rank Fusion
        """

        # -------------------------------------------------
        # 1. BM25 retrieval
        # -------------------------------------------------

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=retrieval_k
        )

        # -------------------------------------------------
        # 2. Dense retrieval
        # -------------------------------------------------

        query_embedding = self.embedder.embed_text(query)

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=retrieval_k
        )

        # -------------------------------------------------
        # 3. Store RRF scores
        # -------------------------------------------------

        fused_scores = {}
        chunk_data = {}

        # -------------------------------------------------
        # BM25 results
        # -------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk = result["chunk"]

            document_id = chunk["document_id"]
            chunk_id = chunk["chunk_id"]

            unique_id = (
                f"{document_id}_chunk_{chunk_id}"
            )

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0)
                + 1 / (rrf_k + rank)
            )

            chunk_data[unique_id] = chunk

        # -------------------------------------------------
        # Dense results
        # -------------------------------------------------

        documents = dense_results.get(
            "documents",
            [[]]
        )

        metadatas = dense_results.get(
            "metadatas",
            [[]]
        )

        distances = dense_results.get(
            "distances",
            [[]]
        )

        dense_documents = (
            documents[0]
            if documents
            else []
        )

        dense_metadatas = (
            metadatas[0]
            if metadatas
            else []
        )

        dense_distances = (
            distances[0]
            if distances
            else []
        )

        for rank, metadata in enumerate(
            dense_metadatas,
            start=1
        ):

            document_id = metadata["document_id"]
            chunk_id = metadata["chunk_id"]

            unique_id = (
                f"{document_id}_chunk_{chunk_id}"
            )

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0)
                + 1 / (rrf_k + rank)
            )

            if unique_id not in chunk_data:

                document_index = rank - 1

                document_text = dense_documents[
                    document_index
                ]

                chunk_data[unique_id] = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "page_number": metadata[
                        "page_number"
                    ],
                    "text": document_text
                }

        # -------------------------------------------------
        # 4. Sort by RRF score
        # -------------------------------------------------

        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # -------------------------------------------------
        # 5. Return top results
        # -------------------------------------------------

        results = []

        for unique_id, score in ranked_chunks[:top_k]:

            results.append({
                "chunk": chunk_data[unique_id],
                "score": score
            })

        return results
