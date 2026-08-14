import os

from dotenv import load_dotenv
from google import genai

from backend.generation.llm import LLM


class GeminiLLM(LLM):
    """
    LLM implementation using Google Gemini.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash"
    ):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response using Gemini Interactions API.
        """

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt
        )

        return interaction.output_text
