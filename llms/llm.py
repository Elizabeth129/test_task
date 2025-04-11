# Copyright (C) 2025 Yelyzaveta Ivanytska

from typing import List
from abc import ABC, abstractmethod
from langchain.schema import Document

class Llm(ABC):
    @abstractmethod
    def search_notes(self, docs: List[Document], query: str) -> str:
        """Return answer based on notes."""