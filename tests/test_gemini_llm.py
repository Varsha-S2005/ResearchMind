from backend.generation.gemini_llm import GeminiLLM


def test_gemini_llm():

    llm = GeminiLLM()

    prompt = """
    Explain federated learning in simple terms
    in two sentences.
    """

    response = llm.generate(prompt)

    print("\nGemini Response:")
    print(response)

    assert isinstance(response, str)
    assert len(response) > 0

    print("\nGemini LLM test passed!")


if __name__ == "__main__":
    test_gemini_llm()
