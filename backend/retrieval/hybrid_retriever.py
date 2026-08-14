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

        # BM25 ranking
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk"]["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            chunk_data[chunk_id] = result["chunk"]

        # Dense ranking
        for rank, (document, metadata) in enumerate(
            zip(
                dense_results["documents"][0],
                dense_results["metadatas"][0]
            ),
            start=1
        ):
            chunk_id = metadata["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_number": metadata["page_number"],
                    "text": document
                }

        # -------------------------
        # 4. Sort by fused score
        # -------------------------
        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # -------------------------
        # 5. Return top results
        # -------------------------
        results = []

        for chunk_id, score in ranked_chunks[:top_k]:
            results.append({
                "chunk": chunk_data[chunk_id],
                "score": score
            })

        return results
