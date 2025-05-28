from abc import ABC, abstractmethod
from typing import Any

class ReasoningProcess(ABC):
    """
    An abstract base class for defining different reasoning processes.
    """

    @abstractmethod
    def execute(self, problem_description: str, model_name: str, *args, **kwargs) -> None:
        """
        Executes the reasoning process.

        Args:
            problem_description: The description of the problem to be solved.
            model_name: The name of the model to be used for reasoning.
            *args: Additional positional arguments for the specific process.
            **kwargs: Additional keyword arguments for the specific process.
        """
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """
        Returns the final result of the reasoning process.

        Returns:
            The result of the reasoning process, which can be of any type.
        """
        pass
