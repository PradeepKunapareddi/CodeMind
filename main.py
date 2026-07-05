# ============================================================
# STEP 4 — FASTAPI BACKEND
#
# This is the REST API layer of CodeMind.
# It exposes two endpoints:
#
# POST /index   → Load code files and store in ChromaDB
# POST /ask     → Ask a question about the codebase
# GET  /health  → Check if the API is running
#
# Run with: uvicorn main:app --reload
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from file_loader import load_code_files
from vector_store import create_vector_store, load_existing_vector_store
from rag_chain import create_rag_chain, ask_question

# ── APP SETUP ─────────────────────────────────────────────────
app = FastAPI(
    title="CodeMind — AI Codebase Q&A Assistant",
    description="Ask any question about your codebase in plain English",
    version="1.0.0"
)

# Global variable to store the RAG chain
# Once initialized, it stays in memory for all requests
rag_chain = None

# ── GET GROQ API KEY ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────
class IndexRequest(BaseModel):
    folder_path: str  # Path to your code folder

    class Config:
        json_schema_extra = {
            "example": {
                "folder_path": "./sample_code"
            }
        }

class QuestionRequest(BaseModel):
    question: str  # The question to ask about the codebase

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How does the loan prediction API work?"
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def home():
    """
    Home endpoint — just confirms the API is running
    """
    return {
        "message": "🧠 CodeMind is running!",
        "usage": {
            "step1": "POST /index with folder_path to load your code",
            "step2": "POST /ask with your question"
        }
    }


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "rag_chain_ready": rag_chain is not None
    }


@app.post("/index")
def index_codebase(request: IndexRequest):
    """
    STEP 1: Load all code files from a folder and store in ChromaDB.

    What happens:
    1. Reads all .py .js .java .md files from the folder
    2. Splits them into chunks
    3. Creates embeddings using Sentence Transformers
    4. Stores everything in ChromaDB

    Only needs to be called ONCE per codebase.
    After this, just use /ask to query.
    """
    global rag_chain

    # Check if folder exists
    if not os.path.exists(request.folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {request.folder_path}"
        )

    try:
        # Step 1: Load code files
        print(f"\n📂 Loading files from: {request.folder_path}")
        documents = load_code_files(request.folder_path)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No code files found in the folder"
            )

        # Step 2: Create vector store
        print("\n🧠 Creating embeddings and storing in ChromaDB...")
        vector_store = create_vector_store(documents)

        # Step 3: Create RAG chain
        print("\n⛓️  Building RAG chain with Groq LLM...")
        rag_chain = create_rag_chain(vector_store, GROQ_API_KEY)

        return {
            "message": "✅ Codebase indexed successfully!",
            "files_loaded": len(documents),
            "folder": request.folder_path,
            "status": "Ready to answer questions. Use POST /ask"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    STEP 2: Ask a question about the indexed codebase.

    What happens:
    1. Your question goes to ChromaDB
    2. ChromaDB finds the most relevant code chunks
    3. Groq LLM reads the chunks and generates an answer
    4. You get the answer + which files were used

    Example questions:
    - "How does the loan prediction work?"
    - "What API endpoints are available?"
    - "How is the model loaded?"
    - "What does the predict function do?"
    """
    global rag_chain

    # Check if codebase has been indexed first
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed yet. Please call POST /index first."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = ask_question(rag_chain, request.question)
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-existing")
def load_existing():
    """
    Load an already indexed ChromaDB vector store.
    Use this if you have already run /index before
    and the ChromaDB data is already saved on disk.
    """
    global rag_chain

    try:
        vector_store = load_existing_vector_store()
        rag_chain = create_rag_chain(vector_store, GROQ_API_KEY)

        return {
            "message": "✅ Existing vector store loaded!",
            "status": "Ready to answer questions. Use POST /ask"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
