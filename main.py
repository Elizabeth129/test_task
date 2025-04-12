# Copyright (C) 2025 Yelyzaveta Ivanytska

import os
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from llms.gpt.gpt_llm import GptLlm
from models.note_read import NoteRead
from models.note_create import NoteCreate
from langchain_openai import OpenAIEmbeddings
from vectorstore.chroma import ChromaVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectore_store = ChromaVectorStore(os.getenv("CHROMA_COLLECTION"), embeddings, os.getenv("CHROMA_PATH"))
llm = GptLlm("gpt-4o-mini", model_provider="openai", api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


@app.post("/notes/")
def create_note(note: NoteCreate):
    metadata = vectore_store.add_note_to_store(note.title, note.content)
    return {"message": "Note saved", "metadata": metadata}


@app.get("/notes/recent/", response_model=List[NoteRead])
def recent_notes(n: int = 5):
    notes = vectore_store.get_recent_notes(n)
    return [{
        "title": meta["title"],
        "content": content,
        "created_at": meta["created_at"]
    } for meta, content in notes]


@app.get("/notes/search/")
def search(query: str, k: int):
    docs = vectore_store.storage.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found."
    
    answer = llm.search_notes(docs, query)
    return {"answer": answer}