class HybridRetriever:
    """
    Hybrid retrieval using:
    1. BM25 keyword retrieval
    2. Dense semantic retrieval
    3. Weighted Reciprocal Rank Fusion (RRF)

    Dense retrieval is given slightly higher weight because
    ResearchMind is primarily used for semantic research questions.
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
        retrieval_k: int = 40,
        rrf_k: int = 60
    ) -> list[dict]:

        # =====================================================
        # 1. BM25 RETRIEVAL
        # =====================================================

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=retrieval_k
        )

        # =====================================================
        # 2. DENSE RETRIEVAL
        # =====================================================

        query_embedding = self.embedder.embed_text(
            query
        )

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=retrieval_k
        )

        # =====================================================
        # 3. STORAGE FOR FUSION
        # =====================================================

        fused_scores = {}
        chunk_data = {}

        # -----------------------------------------------------
        # Weights
        #
        # Dense retrieval gets slightly higher weight because
        # semantic similarity is important for research queries.
        # -----------------------------------------------------

        BM25_WEIGHT = 0.4
        DENSE_WEIGHT = 0.6

        # =====================================================
        # 4. ADD BM25 RESULTS
        # =====================================================

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

            rrf_score = (
                BM25_WEIGHT
                /
                (rrf_k + rank)
            )

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0.0)
                +
                rrf_score
            )

            chunk_data[unique_id] = chunk

        # =====================================================
        # 5. ADD DENSE RESULTS
        # =====================================================

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

            rrf_score = (
                DENSE_WEIGHT
                /
                (rrf_k + rank)
            )

            fused_scores[unique_id] = (
                fused_scores.get(unique_id, 0.0)
                +
                rrf_score
            )

            # -------------------------------------------------
            # If BM25 did not retrieve this chunk, reconstruct
            # it from ChromaDB metadata and document text.
            # -------------------------------------------------

            if unique_id not in chunk_data:

                document_index = rank - 1

                if document_index < len(
                    dense_documents
                ):

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

        # =====================================================
        # 6. SORT FUSED RESULTS
        # =====================================================

        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # =====================================================
        # 7. RETURN TOP-K
        # =====================================================

        results = []

        for unique_id, score in ranked_chunks[:top_k]:

            results.append({
                "chunk": chunk_data[unique_id],
                "score": float(score)
            })

        return results
