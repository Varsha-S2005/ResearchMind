from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker for ResearchMind.

    Hybrid retrieval first produces a candidate pool.

    The cross-encoder reranks the strongest hybrid candidates.
    Hybrid and cross-encoder rankings are combined using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        rrf_k: int = 60,
        rerank_k: int = 40
    ):
        self.model = CrossEncoder(model_name)

        self.rrf_k = rrf_k
        self.rerank_k = rerank_k

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not results:
            return []

        candidates = results[:self.rerank_k]

        pairs = [
            [
                query,
                result["chunk"]["text"]
            ]
            for result in candidates
        ]

        cross_scores = self.model.predict(
            pairs
        )

        cross_ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: float(
                cross_scores[index]
            ),
            reverse=True
        )

        cross_ranks = {}

        for rank, index in enumerate(
            cross_ranked_indices,
            start=1
        ):
            cross_ranks[index] = rank

        reranked = []

        for index, result in enumerate(candidates):

            hybrid_rank = index + 1
            cross_rank = cross_ranks[index]

            hybrid_rrf = (
                1.0 /
                (
                    self.rrf_k +
                    hybrid_rank
                )
            )

            cross_rrf = (
                1.0 /
                (
                    self.rrf_k +
                    cross_rank
                )
            )

            combined_score = (
                0.7 * hybrid_rrf +
                0.3 * cross_rrf
            )

            reranked.append({
                "chunk": result["chunk"],

                "retrieval_score": float(
                    result.get(
                        "score",
                        0.0
                    )
                ),

                "rerank_score": float(
                    cross_scores[index]
                ),

                "hybrid_rank": hybrid_rank,

                "cross_encoder_rank": cross_rank,

                "score": float(
                    combined_score
                )
            })

        reranked.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return reranked[:top_k]
