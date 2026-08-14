from backend.generation.llm import LLM


class MockLLM(LLM):
    """
    Mock LLM used for testing the generation pipeline.
    """

    def generate(self, prompt: str) -> str:
        return (
            "This is a mock response generated from the "
            "provided research context."
        )
