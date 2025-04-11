# Copyright (C) 2025 Yelyzaveta Ivanytska

from typing import List
from llms.llm import Llm
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from langchain.chains.question_answering import load_qa_chain
from langchain_core.language_models.chat_models import BaseChatModel

class GptLlm(Llm):

    llm_model: BaseChatModel

    def __init__(self, model: str, model_provider: str, api_key: str) -> None:
        self.llm_model = init_chat_model(model, model_provider=model_provider, api_key=api_key)

    def search_notes(self, docs: List[Document], query: str) -> str:
        chain = load_qa_chain(self.llm_model, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response
        


    
