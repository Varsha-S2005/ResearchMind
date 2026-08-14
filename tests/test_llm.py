from backend.generation.mock_llm import MockLLM


def test_llm():

    llm = MockLLM()

    prompt = """
    Context:
    Federated learning allows multiple vehicles
    to collaboratively train a model.

    Question:
    What is federated learning?
    """

    response = llm.generate(prompt)

    print("\nLLM Response:")
    print(response)

    assert isinstance(response, str)
    assert len(response) > 0

    print("\nLLM interface test passed!")


if __name__ == "__main__":
    test_llm()
