# Copyright (C) 2025 Yelyzaveta Ivanytska

from pydantic import BaseModel

class NoteCreate(BaseModel):
    title: str
    content: str