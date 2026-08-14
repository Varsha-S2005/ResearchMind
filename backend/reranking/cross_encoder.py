from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not results:
            return []

        # -------------------------------------------------
        # Prepare query-document pairs
        # -------------------------------------------------

        pairs = [
            [
                query,
                result["chunk"]["text"]
            ]
            for result in results
        ]

        # -------------------------------------------------
        # Cross-encoder scores
        # -------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            reranked.append({
                "chunk": result["chunk"],

                # Original hybrid score
                "retrieval_score": result["score"],

                # Cross-encoder score
                "rerank_score": float(score)
            })

        # -------------------------------------------------
        # Sort using cross-encoder relevance
        # -------------------------------------------------

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True
        )

        # -------------------------------------------------
        # Return final top-k
        # -------------------------------------------------

        final_results = []

        for result in reranked[:top_k]:

            final_results.append({
                "chunk": result["chunk"],
                "score": result["rerank_score"]
            })

        return final_results
