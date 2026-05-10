# base agent for all agents to inherit 

from abc import ABC, abstractmethod
from typing import Any
from src.kg.schema import GraphUpdate


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, input_data: dict[str, Any]) -> GraphUpdate:
        raise NotImplementedError