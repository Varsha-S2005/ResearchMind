import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retrieval using the BM25 algorithm.

    BM25 is useful for exact terminology and technical keywords,
    while dense retrieval handles semantic similarity.
    """

    def __init__(self, chunks: list[dict]):

        self.chunks = chunks

        # Tokenize every document chunk
        tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized tokens.

        The tokenizer:
        1. Converts text to lowercase.
        2. Separates punctuation.
        3. Preserves technical terms such as:
           non-IID
           FL-IDS
           V2X
           CAN
        4. Removes unnecessary punctuation.
        """

        text = text.lower()

        # Normalize common separators
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace("/", " ")
        
        # Keep letters, numbers and hyphens.
        text = re.sub(
            r"[^a-z0-9\-]+",
            " ",
            text
        )

        tokens = text.split()

        # Remove empty tokens
        tokens = [
            token
            for token in tokens
            if token
        ]

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Return the top-k chunks ranked by BM25 score.
        """

        if not self.chunks:
            return []

        # Tokenize query using the same preprocessing
        # applied to the document chunks.
        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        # Calculate BM25 relevance scores
        scores = self.bm25.get_scores(
            query_tokens
        )

        # Rank chunks by score
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
