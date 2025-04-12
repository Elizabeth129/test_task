# 📝 Personal Note-Taking App

This is an MVP for a personal note-taking application that allows you to:

- Save flexible text notes
- Retrieve your most recent notes
- Ask questions in natural language and receive answers using RAG powered by LLM

## 🚀 Features

✅ Save notes with title and content  
✅ View the latest N notes  
✅ Ask natural questions — get smart, LLM-generated answers from your notes  
✅ All notes stored using vector embeddings in Chroma  
✅ No SQL database required

Built using:
- 🧠 LangChain (LLMs + RAG)
- ⚡ FastAPI (REST API)
- 🧪 ChromaDB (vector store for semantic search)
- 🔑 OpenAI (LLM & Embeddings)

## 📦 Installation & Running Locally

Follow these steps to set up and run the app on your local machine.

### 1. Clone the Repository

Clone the repository to your local machine using Git

### 2. Set Up a Python Virtual Environment

Create and activate a virtual environment to isolate dependencies

On macOS/Linux:   
    ``` python -m venv venv ```
    ``` source venv/bin/activate ```

On Windows:  
    ``` python -m venv venv ```  
    ``` venv\Scripts\activate ```

### 3. Install Dependencies

Install all required Python packages using pip:  
    ```pip install -r requirements.txt```  

### 4. Set up .env file

 - Add .env file to the project root folder. 
 - Add required variables: 
    ``` CHROMA_COLLECTION = "notes" ```
    ``` CHROMA_PATH = "./chroma_notes" ```
    ``` OPENAI_API_KEY = your_key ```
    
    where:    
        - CHROMA_PATH - filesystem path where Chroma will store the vector data persistently  
        - CHROMA_COLLECTION - name of the collection inside the vector store (like a "table" in a DB)  
        - OPENAI_API_KEY - key for access OpenAI

### 5. Start the Application

    Run the FastAPI application using Uvicorn:  
    ``` uvicorn main:app --reload ``` 

### 6. Access the application 

    Visit the Swagger API documentation in your browser:
        http://localhost:8000/docs

    You can use the interactive API interface to test endpoints such as:  
    - POST /notes/ to add a note.
    - GET /notes/recent/ to fetch recent notes.
    - GET /notes/search/ to ask questions and retrieve answers.