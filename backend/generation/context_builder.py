class ContextBuilder:
    """
    Builds the context that will be provided to the LLM
    from retrieved and reranked chunks.
    """

    def build(
        self,
        results: list[dict]
    ) -> str:
        """
        Build a formatted research context from
        retrieved chunks.
        """

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            document_id = chunk.get(
                "document_id",
                "Unknown document"
            )

            page_number = chunk.get(
                "page_number",
                "Unknown"
            )

            chunk_id = chunk.get(
                "chunk_id",
                "Unknown"
            )

            score = result.get(
                "score",
                0
            )

            text = chunk.get(
                "text",
                ""
            )

            context_parts.append(
                f"Source {index}\n"
                f"Document: {document_id}\n"
                f"Page: {page_number}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Score: {score}\n"
                f"Content:\n{text}"
            )

        return "\n\n".join(
            context_parts
        )
