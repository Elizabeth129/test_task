# Copyright (C) 2025 Yelyzaveta Ivanytska

from typing import List
from llms.llm import Llm
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.combine_documents import create_stuff_documents_chain

APPOINTMENTS_PROMPT = """You are an intelligent assistant that extracts structured data about appointments 
from text notes. Each note may contain information about one or more appointments or may not. 
For each appointment you find, extract the following fields:
    Title: A short summary of the appointment (e.g., "Doctor appointment", "Team meeting").
    Date: The date of the appointment in YYYY-MM-DD format, if available.
    Time: The time of the appointment in 24-hour format (e.g., "14:30"), if available.
    Location: Where the appointment is taking place (e.g., "Room 405", "Downtown Clinic"), if mentioned.
    People: Names of people involved, if mentioned.
    Note_Source: The original text where this appointment was found.

If a field is not mentioned, leave it as null. Return the result as a list of JSON objects, one per appointment.

Here are the text notes:
{context}
"""

TASKS_PROMPT = """You are an intelligent assistant that extracts structured data about tasks from text notes. 
Each note may contain one or more tasks or may not. Do not include appointments into tasks list. For each task you find, extract the following fields:
    Task_Title: A short description of the task (e.g., "Buy groceries", "Prepare presentation").
    Due_Date: The due date of the task in YYYY-MM-DD format, if available.
    Priority: The priority of the task (e.g., "High", "Medium", "Low"), if available.
    Status: The current status of the task (e.g., "Pending", "Completed", "In Progress"), if available.
    Assignee: The person responsible for the task, if mentioned.
    Note_Source: The original text where this task was found.

If a field is not mentioned, leave it as null. Return the result as a list of JSON objects, one per task.

Here are the text notes:
{context}
"""

RECIPES_PROMPT = """You are an intelligent assistant that extracts structured data about recipes from text notes. 
Each note may contain one or more recipes or may not. For each recipe you find, extract the following fields:
    Recipe_Title: The name of the recipe (e.g., "Spaghetti Carbonara", "Chicken Salad").
    Ingredients: A list of ingredients required for the recipe (e.g., ["pasta", "eggs", "cheese", "bacon"]).
    Instructions: A step-by-step list of instructions for preparing the dish (e.g., ["Boil the pasta", "Cook the bacon", "Mix eggs and cheese"]).
    Preparation Time: The time required to prepare the dish (e.g., "30 minutes"), if available.
    Servings: The number of servings the recipe makes, if available.
    Note_Source: The original text where this recipe was found.

If a field is not mentioned, leave it as null. Return the result as a list of JSON objects, one per recipe.

Here are the text notes:
{context}
"""

VOCABULARY_PROMPT = """You are an intelligent assistant that extracts structured data about vocabulary words 
from text notes.
Note: vocabulary word - only where translation mentioned.
Each note may contain one or more vocabulary words or may not. For each vocabulary word you find, 
extract the following fields:
    Word: The vocabulary word (e.g., "benevolent", "arduous").
    Definition: The definition of the word (e.g., "Showing kindness or goodwill", "Involving or requiring strenuous effort").
    Part_of_Speech: The part of speech (e.g., "Adjective", "Noun", "Verb").
    Example_Sentence: A sentence that demonstrates the word used in context, if available.
    Note_Source: The original text where this word was found.

If a field is not mentioned, leave it as null. Return the result as a list of JSON objects, one per vocabulary word.

Here are the text notes:
{context}
"""

class GptLlm(Llm):

    llm_model: BaseChatModel

    def __init__(self, model: str, model_provider: str, api_key: str) -> None:
        self.llm_model = init_chat_model(model, model_provider=model_provider, api_key=api_key)


    def search_notes(self, docs: List[Document], query: str) -> str:
        chain = load_qa_chain(self.llm_model, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return response
    

    def extract_appointments(self, docs: List[Document]) -> str:
        prompt = ChatPromptTemplate.from_template(APPOINTMENTS_PROMPT)
        chain = create_stuff_documents_chain(self.llm_model, prompt)
        result = chain.invoke({"context": docs})
        return result
    

    def extract_tasks(self, docs: List[Document]) -> str:
        prompt = ChatPromptTemplate.from_template(TASKS_PROMPT)
        chain = create_stuff_documents_chain(self.llm_model, prompt)
        result = chain.invoke({"context": docs})
        return result
    

    def extract_recipes(self, docs: List[Document]) -> str:
        prompt = ChatPromptTemplate.from_template(RECIPES_PROMPT)
        chain = create_stuff_documents_chain(self.llm_model, prompt)
        result = chain.invoke({"context": docs})
        return result
    

    def extract_vocabulary(self, docs: List[Document]) -> str:
        prompt = ChatPromptTemplate.from_template(VOCABULARY_PROMPT)
        chain = create_stuff_documents_chain(self.llm_model, prompt)
        result = chain.invoke({"context": docs})
        return result

