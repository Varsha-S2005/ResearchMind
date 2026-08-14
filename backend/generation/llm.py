from abc import ABC, abstractmethod


class LLM(ABC):
    """
    Abstract interface for a Large Language Model.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response from a prompt.
        """
        pass
