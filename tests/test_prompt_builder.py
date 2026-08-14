from backend.generation.prompt_builder import PromptBuilder


def test_prompt_builder():

    question = "What are the challenges of federated learning?"

    context = """
[Source 1]
Page: 6
Chunk ID: 12
Text:
Data heterogeneity is an important challenge in federated learning.

[Source 2]
Page: 7
Chunk ID: 14
Text:
Many studies use unrealistic IID data distributions.
"""

    builder = PromptBuilder()

    prompt = builder.build(
        question=question,
        context=context
    )

    print("\nGenerated Prompt:\n")
    print(prompt)

    assert question in prompt
    assert context in prompt
    assert "ONLY" in prompt
    assert "Do not invent facts" in prompt

    print("\nPrompt builder test passed!")


if __name__ == "__main__":
    test_prompt_builder()
