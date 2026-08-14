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
        """
        Rerank retrieved chunks based on query-document relevance.
        """

        if not results:
            return []

        pairs = [
            [query, result["chunk"]["text"]]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            reranked.append({
                "chunk": result["chunk"],
                "score": float(score)
            })

        reranked.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return reranked[:top_k]
