"""Generic execution contract: analysts and signal builders return different types."""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

AgentResult = TypeVar("AgentResult")

class BaseAgent(ABC, Generic[AgentResult]):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, input_data: dict[str, Any]) -> AgentResult:
        raise NotImplementedError
