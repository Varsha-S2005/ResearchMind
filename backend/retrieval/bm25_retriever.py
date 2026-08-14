from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retrieval using the BM25 algorithm.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        # Tokenize every chunk
        tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into lowercase word tokens.
        """
        return text.lower().split()

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Return the top-k chunks ranked by BM25 score.
        """

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "chunk": self.chunks[index],
                "score": float(scores[index])
            })

        return results
