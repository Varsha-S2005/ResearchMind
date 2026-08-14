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
        retrieval_k: int = 20,
        rrf_k: int = 60
    ) -> list[dict]:

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
        # 3. Prepare RRF data
        # -------------------------------------------------

        fused_scores = {}
        chunk_data = {}

        # -------------------------------------------------
        # 4. Add BM25 results
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

            rrf_score = 1 / (rrf_k + rank)

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0)
                + rrf_score
            )

            chunk_data[unique_id] = chunk

        # -------------------------------------------------
        # 5. Add dense results
        # -------------------------------------------------

        documents = dense_results.get(
            "documents",
            [[]]
        )

        metadatas = dense_results.get(
            "metadatas",
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

        for rank, metadata in enumerate(
            dense_metadatas,
            start=1
        ):

            document_id = metadata["document_id"]
            chunk_id = metadata["chunk_id"]

            unique_id = (
                f"{document_id}_chunk_{chunk_id}"
            )

            rrf_score = 1 / (rrf_k + rank)

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0)
                + rrf_score
            )

            # If BM25 did not retrieve this chunk,
            # construct the chunk from dense metadata.
            if unique_id not in chunk_data:

                document_index = rank - 1

                chunk_data[unique_id] = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "page_number": metadata[
                        "page_number"
                    ],
                    "text": dense_documents[
                        document_index
                    ]
                }

        # -------------------------------------------------
        # 6. Sort using RRF score
        # -------------------------------------------------

        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # -------------------------------------------------
        # 7. Return top-k
        # -------------------------------------------------

        results = []

        for unique_id, score in ranked_chunks[:top_k]:

            results.append({
                "chunk": chunk_data[unique_id],
                "score": float(score)
            })

        return results
