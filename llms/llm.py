# Copyright (C) 2025 Yelyzaveta Ivanytska

from typing import List
from abc import ABC, abstractmethod
from langchain.schema import Document

class Llm(ABC):
    @abstractmethod
    def search_notes(self, docs: List[Document], query: str) -> str:
        """Return answer based on notes."""

    @abstractmethod
    def extract_appointments(self, docs: List[Document]) -> str:
        """Return information about appointments based on notes."""

    @abstractmethod
    def extract_tasks(self, docs: List[Document]) -> str:
        """Return information about tasks based on notes."""

    @abstractmethod
    def extract_recipes(self, docs: List[Document]) -> str:
        """Return recipes from notes."""

    @abstractmethod
    def extract_vocabulary(self, docs: List[Document]) -> str:
        """Return vocabulary from notes."""