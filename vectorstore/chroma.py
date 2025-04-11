# Copyright (C) 2025 Yelyzaveta Ivanytska

import uuid
from typing import Any
from datetime import datetime
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from vectorstore.base import VectorStoreABC

class ChromaVectorStore(VectorStoreABC):

    storage: Chroma
    """Storage instance"""

    def __init__(self, collection_name, embedding_function, persist_directory):
        self.storage = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

    def add_note_to_store(self, title: str, content: str) -> dict[str, Any]:
        note_id = str(uuid.uuid4())
        metadata = {
            "id": note_id,
            "title": title,
            "created_at": datetime.now().isoformat()
        }
        doc = Document(page_content=content, metadata=metadata)
        self.storage.add_documents([doc])
        self.storage.persist()
        return metadata

    def get_recent_notes(self, n: int) -> list[tuple[Any, Any]]:
        all = self.storage.get()
        docs = sorted(zip(all['metadatas'], all['documents']), key=lambda x: x[0]['created_at'], reverse=True)
        return docs[:n]