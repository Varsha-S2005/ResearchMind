from backend.generation.context_builder import ContextBuilder
from backend.generation.prompt_builder import PromptBuilder


class RAGPipeline:
    """
    Coordinates retrieval, reranking, context construction,
    prompt construction, LLM generation, and grounding verification.

    Pipeline:

        Hybrid retrieval
            ↓
        Cross-encoder reranking
            ↓
        Context construction
            ↓
        Gemini answer generation
            ↓
        Grounding critic verification
            ↓
        Final response
    """

    def __init__(
        self,
        retriever,
        reranker,
        llm,
        grounding_critic,
        retrieval_k: int = 40
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.grounding_critic = grounding_critic

        self.retrieval_k = retrieval_k

        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()

    def answer(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:
        """
        Generate an answer and verify its grounding
        against the retrieved research evidence.
        """

        # -------------------------------------------------
        # 1. Hybrid retrieval
        # -------------------------------------------------

        retrieved_results = self.retriever.search(
            question,
            top_k=self.retrieval_k
        )

        # -------------------------------------------------
        # 2. Cross-encoder reranking
        # -------------------------------------------------

        reranked_results = self.reranker.rerank(
            question,
            retrieved_results,
            top_k=top_k
        )

        # -------------------------------------------------
        # 3. Build research context
        # -------------------------------------------------

        context = self.context_builder.build(
            reranked_results
        )

        # -------------------------------------------------
        # 4. Build grounded prompt
        # -------------------------------------------------

        prompt = self.prompt_builder.build(
            question=question,
            context=context
        )

        # -------------------------------------------------
        # 5. Generate answer
        # -------------------------------------------------

        answer = self.llm.generate(
            prompt
        )

        # -------------------------------------------------
        # 6. Grounding verification
        # -------------------------------------------------

        critic_chunks = [
            result["chunk"]
            for result in reranked_results
        ]

        verification = self.grounding_critic.verify(
            question=question,
            answer=answer,
            chunks=critic_chunks
        )

        # -------------------------------------------------
        # 7. Build source information
        # -------------------------------------------------

        sources = []

        for result in reranked_results:

            chunk = result["chunk"]

            sources.append({
                "source_id": f"Source {len(sources) + 1}",
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "score": result.get("score", 0)
            })

        # -------------------------------------------------
        # 8. Return final result
        # -------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "verification": verification
        }
