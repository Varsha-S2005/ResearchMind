from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker for ResearchMind.

    Hybrid retrieval first produces a candidate pool.

    The cross-encoder is intentionally applied only to the
    top `rerank_k` hybrid candidates. This prevents noisy
    low-ranked candidates from jumping directly into the
    final Top-K.

    Hybrid and cross-encoder rankings are combined using RRF.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        rrf_k: int = 60,
        rerank_k: int = 40
    ):
        self.model = CrossEncoder(model_name)

        self.rrf_k = rrf_k

        # Only rerank the strongest hybrid candidates.
        self.rerank_k = rerank_k

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not results:
            return []

        # -------------------------------------------------
        # 1. Keep only the top hybrid candidates
        # -------------------------------------------------

        candidates = results[:self.rerank_k]

        # -------------------------------------------------
        # 2. Prepare query-document pairs
        # -------------------------------------------------

        pairs = [
            [
                query,
                result["chunk"]["text"]
            ]
            for result in candidates
        ]

        # -------------------------------------------------
        # 3. Cross-encoder prediction
        # -------------------------------------------------

        cross_scores = self.model.predict(
            pairs
        )

        # -------------------------------------------------
        # 4. Cross-encoder ranking
        # -------------------------------------------------

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

        # -------------------------------------------------
        # 5. Hybrid ranking
        #
        # Since candidates are results[:rerank_k],
        # their positions already represent their
        # original hybrid ranking.
        # -------------------------------------------------

        reranked = []

        for index, result in enumerate(candidates):

            hybrid_rank = index + 1

            cross_rank = cross_ranks[index]

            # -------------------------------------------------
            # RRF
            # -------------------------------------------------

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
                hybrid_rrf +
                cross_rrf
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

        # -------------------------------------------------
        # 6. Sort using RRF score
        # -------------------------------------------------

        reranked.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        # -------------------------------------------------
        # 7. Return final Top-K
        # -------------------------------------------------

        return reranked[:top_k]
