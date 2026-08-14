class HybridRetriever:
    """
    Combines BM25 and dense retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, bm25_retriever, vector_store, embedder):
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

        # -------------------------
        # 1. BM25 retrieval
        # -------------------------
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=retrieval_k
        )

        # -------------------------
        # 2. Dense retrieval
        # -------------------------
        query_embedding = self.embedder.embed_text(query)

        dense_results = self.vector_store.search(
            query_embedding,
            top_k=retrieval_k
        )

        # -------------------------
        # 3. Store RRF scores
        # -------------------------
        fused_scores = {}
        chunk_data = {}

        # -------------------------
        # 4. BM25 ranking
        # -------------------------
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):
            chunk = result["chunk"]

            document_id = chunk.get(
                "document_id",
                "default_document"
            )

            chunk_id = chunk["chunk_id"]

            result_id = f"{document_id}_chunk_{chunk_id}"

            fused_scores[result_id] = (
                fused_scores.get(result_id, 0)
                + 1 / (rrf_k + rank)
            )

            chunk_data[result_id] = {
                **chunk,
                "document_id": document_id,
                "filename": chunk.get(
                    "filename",
                    "unknown.pdf"
                )
            }

        # -------------------------
        # 5. Dense ranking
        # -------------------------
        documents = dense_results.get(
            "documents",
            [[]]
        )[0]

        metadatas = dense_results.get(
            "metadatas",
            [[]]
        )[0]

        for rank, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1
        ):
            document_id = metadata.get(
                "document_id",
                "default_document"
            )

            chunk_id = metadata["chunk_id"]

            result_id = f"{document_id}_chunk_{chunk_id}"

            fused_scores[result_id] = (
                fused_scores.get(result_id, 0)
                + 1 / (rrf_k + rank)
            )

            if result_id not in chunk_data:
                chunk_data[result_id] = {
                    "document_id": document_id,
                    "filename": metadata.get(
                        "filename",
                        "unknown.pdf"
                    ),
                    "chunk_id": chunk_id,
                    "page_number": metadata["page_number"],
                    "text": document
                }

        # -------------------------
        # 6. Sort by RRF score
        # -------------------------
        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # -------------------------
        # 7. Return top results
        # -------------------------
        results = []

        for result_id, score in ranked_chunks[:top_k]:
            results.append({
                "chunk": chunk_data[result_id],
                "score": score
            })

        return results
