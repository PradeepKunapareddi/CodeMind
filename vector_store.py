# ============================================================
# EMBEDDINGS + CHROMADB VECTOR STORE
#
# What happens here:
# 1. Take the code files loaded in Step 1
# 2. Split them into smaller chunks
# 3. Convert each chunk into embeddings using Sentence Transformers
# 4. Store all embeddings in ChromaDB
#
# Why chunking?
# A file may have 500 lines — we cannot send all 500 lines
# to the LLM at once. So we split into smaller pieces (chunks)
# and only send the most relevant chunk when a question is asked.
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os

# ── CONFIGURATION ────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"        # Where ChromaDB stores data on disk
COLLECTION_NAME = "codemind"          # Name of our collection inside ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Free Sentence Transformer model from HuggingFace
                                       # Small, fast, and works well for code

# ── CHUNK SETTINGS ───────────────────────────────────────────
# chunk_size = how many characters per chunk
# chunk_overlap = how many characters overlap between chunks
#                 overlap helps maintain context at chunk boundaries
CHUNK_SIZE    = 1000 
CHUNK_OVERLAP = 200 


def create_vector_store(documents: list[dict]) -> Chroma:
    """
    Takes loaded code files, splits them into chunks,
    creates embeddings, and stores in ChromaDB.

    Input:  List of {"filename": ..., "content": ...}
    Output: ChromaDB vector store ready for searching
    """

    print("Converting files to LangChain Document format...")

    # Convert our dict format to LangChain Document format
    # LangChain needs Document objects with page_content and metadata
    langchain_docs = []
    for doc in documents:
        langchain_docs.append( 
            Document(
                page_content=doc["content"],
                metadata={"source": doc["filename"]}  # store filename as metadata
            )
        )

    print(f"{len(langchain_docs)} documents ready")

    # ── SPLIT INTO CHUNKS ─────────────────────────────────────
    print("\n Splitting documents into chunks...")

    # RecursiveCharacterTextSplitter splits text by:
    # 1. First tries to split by new lines \n\n
    # 2. Then by single new line \n
    # 3. Then by spaces
    # 4. Then by characters
    # This keeps code structure as intact as possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] 
    )

    chunks = splitter.split_documents(langchain_docs)
    print(f" Created {len(chunks)} chunks from {len(langchain_docs)} files")
    print(f"   Average chunk size: ~{CHUNK_SIZE} characters")

    # ── CREATE EMBEDDINGS ───────────────────────────────────── 
    print("\nLoading Sentence Transformer embedding model...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print("   This converts text into numbers (vectors) that capture meaning")

    # HuggingFace Sentence Transformers — completely FREE
    # all-MiniLM-L6-v2 is a small but powerful model
    # It converts any text into a 384-dimensional vector
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # use CPU (no GPU needed)
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded")

    # ── STORE IN CHROMADB ─────────────────────────────────────
    print("\nStoring embeddings in ChromaDB...")
    print(f"   Database path: {CHROMA_DB_PATH}")

    # Chroma.from_documents does 3 things:
    # 1. Takes each chunk
    # 2. Converts it to embedding using our model
    # 3. Stores chunk + embedding in ChromaDB on disk
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH 
    )

    print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB!")
    print(f"   Database saved at: {CHROMA_DB_PATH}") 

    return vector_store


def load_existing_vector_store() -> Chroma:
    """
    Loads an already existing ChromaDB vector store from disk.
    Use this when you have already processed files before
    and do not want to process them again.
    """
    print("📂 Loading existing ChromaDB vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print("✅ Vector store loaded successfully!")
    return vector_store


def search_similar_chunks(vector_store: Chroma, query: str, k: int = 4) -> list:
    """
    Searches ChromaDB for the most relevant code chunks
    for a given user question.

    k = number of chunks to retrieve (default 4)

    How it works:
    1. Convert the query into an embedding
    2. Compare query embedding with all stored embeddings
    3. Return top k most similar chunks
    """
    print(f"\n🔍 Searching for: '{query}'")
    results = vector_store.similarity_search(query, k=k)
    print(f"✅ Found {len(results)} relevant chunks") 
    return results


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy data
    sample_docs = [
        {
            "filename": "app.py",
            "content": """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Loan Prediction API")
model = joblib.load("loan_model.pkl")

@app.post("/predict")
def predict(data: LoanInput):
    features = np.array([[data.Gender, data.Married,
                          data.ApplicantIncome, data.LoanAmount]])
    prediction = model.predict(features)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    return {"prediction": int(prediction), "result": result}
"""
        }
    ]

    vs = create_vector_store(sample_docs)
    results = search_similar_chunks(vs, "How does loan prediction work?")
    for i, r in enumerate(results):
        print(f"\nChunk {i+1} from {r.metadata['source']}:")
        print(r.page_content[:200])
