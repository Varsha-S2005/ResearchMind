class PromptBuilder:
    """
    Builds prompts for grounded RAG generation.
    """

    def build(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Build a grounded research prompt with source citations.
        """

        return f"""
You are ResearchMind, a research assistant.

Answer the user's question using ONLY the research
context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information,
   clearly say that the available research context is insufficient.
4. Give a concise but informative answer.
5. Every factual claim must be supported by one or more
   sources from the provided research context.
6. Cite sources using their exact source identifier,
   such as [Source 1] or [Source 2].
7. Do not create source numbers that are not present in
   the research context.
8. Place the citation immediately after the claim it supports.
9. Do not mention information that cannot be supported by
   the provided sources.

RESEARCH CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
""".strip()
