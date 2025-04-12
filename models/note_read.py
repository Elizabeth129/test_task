# Copyright (C) 2025 Yelyzaveta Ivanytska

from pydantic import BaseModel

class NoteRead(BaseModel):
    title: str
    content: str
    created_at: str