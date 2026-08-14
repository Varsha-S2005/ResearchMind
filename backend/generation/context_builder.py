class ContextBuilder:
    """
    Builds structured context from reranked document chunks.
    """

    def build(
        self,
        results: list[dict]
    ) -> str:
        """
        Convert reranked chunks into a structured context string.
        """

        if not results:
            return "No relevant research context was found."

        context_parts = []

        for index, result in enumerate(results, start=1):

            chunk = result["chunk"]

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {chunk['filename']}\n"
                f"Document ID: {chunk['document_id']}\n"
                f"Page: {chunk['page_number']}\n"
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Text:\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)
