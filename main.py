# Copyright (C) 2025 Yelyzaveta Ivanytska

import os
import json
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv
from llms.gpt.gpt_llm import GptLlm
from langchain.schema import Document
from models.note_read import NoteRead
from helper import convert_str_to_dict
from models.note_create import NoteCreate
from langchain_openai import OpenAIEmbeddings
from vectorstore.chroma import ChromaVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectore_store = ChromaVectorStore(os.getenv("CHROMA_COLLECTION"), embeddings, os.getenv("CHROMA_PATH"))
llm = GptLlm("gpt-4o-mini", model_provider="openai", api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


@app.post("/notes/")
async def create_note(note: NoteCreate):
    metadata = vectore_store.add_note_to_store(note.title, note.content)
    return {"message": "Note saved", "metadata": metadata}


@app.get("/notes/recent/", response_model=List[NoteRead])
async def recent_notes(n: int = 5):
    notes = vectore_store.get_recent_notes(n)
    return [{
        "title": meta["title"],
        "content": content,
        "created_at": meta["created_at"]
    } for meta, content in notes]


@app.get("/notes/search/")
async def search(query: str, k: int = 5):
    docs = vectore_store.storage.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found."
    
    answer = llm.search_notes(docs, query)
    return {"answer": answer}


@app.get("/notes/extract/appointments/")
async def extract_appointments(k: int = 5):
    docs = vectore_store.get_recent_notes(k)
    answer = llm.extract_appointments([Document(page_content=content, 
                                                metadata={"source": meta["title"], "created_at": meta["created_at"]}) 
                                                for meta, content in docs])
    return convert_str_to_dict(answer)


@app.get("/notes/extract/tasks/")
async def extract_tasks(k: int = 5):
    docs = vectore_store.get_recent_notes(k)
    answer = llm.extract_tasks([Document(page_content=content, 
                                                metadata={"source": meta["title"], "created_at": meta["created_at"]}) 
                                                for meta, content in docs])
    return convert_str_to_dict(answer)


@app.get("/notes/extract/recipes/")
async def extract_recipes(k: int = 5):
    docs = vectore_store.get_recent_notes(k)
    answer = llm.extract_recipes([Document(page_content=content, 
                                                metadata={"source": meta["title"], "created_at": meta["created_at"]}) 
                                                for meta, content in docs])
    return convert_str_to_dict(answer)


@app.get("/notes/extract/vocabulary/")
async def extract_vocabulary(k: int = 5):
    docs = vectore_store.get_recent_notes(k)
    answer = llm.extract_vocabulary([Document(page_content=content, 
                                                metadata={"source": meta["title"], "created_at": meta["created_at"]}) 
                                                for meta, content in docs])
    return convert_str_to_dict(answer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)