# Copyright (C) 2025 Yelyzaveta Ivanytska

from abc import ABC

class VectorStoreABC(ABC):
    def add_note_to_store(self, title: str, content: str):
        """Add new note to the storage."""

    def get_recent_notes(self, n: int):
        """Get last N notes from the storage."""
