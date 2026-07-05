# ============================================================
# STEP 3 — GROQ LLM + RAG CHAIN (Updated for LangChain v1.x)
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
import os
GROQ_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """
You are CodeMind — an intelligent AI assistant that helps developers 
understand codebases by answering questions about the code.

You are given relevant code snippets from a codebase below.
Use ONLY the provided code to answer the question.
If the answer is not in the provided code, say "I could not find this in the codebase."
Always mention which file the code is from when answering.

----- RELEVANT CODE FROM CODEBASE -----
{context}
----------------------------------------

Question: {question}

Answer clearly and explain what the code does in simple terms:
"""


def format_docs(docs):
    """
    Formats retrieved documents into a single string
    with file names for context
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown file")
        content = doc.page_content
        formatted.append(f"FILE: {source}\n{content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(vector_store: Chroma, groq_api_key: str):
    """
    Creates the full RAG chain connecting ChromaDB + Groq LLM.
    Uses modern LangChain v1.x LCEL (LangChain Expression Language)
    """

    print("\n🤖 Setting up Groq LLM...")
    print(f"   Model: {GROQ_MODEL}")

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=GROQ_MODEL,
        temperature=0,
        max_tokens=1024
    )
    print("✅ Groq LLM ready!")

    # Setup retriever
    print("\n🔍 Setting up ChromaDB retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    print("✅ Retriever ready!")

    # Setup prompt
    print("\n📝 Setting up prompt template...")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    print("✅ Prompt ready!")

    # Build RAG chain using LCEL
    print("\n⛓️  Building RAG chain...")
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain ready!")
    print("\n🚀 CodeMind is ready to answer questions!\n")

    return rag_chain, retriever


def ask_question(rag_chain_tuple, question: str) -> dict:
    """
    Takes a user question, runs through RAG chain,
    returns answer with source file references.
    """
    rag_chain, retriever = rag_chain_tuple

    print(f"\n❓ Question: {question}")
    print("⏳ Thinking...")

    # Get source documents
    source_docs = retriever.invoke(question)

    # Get answer
    answer = rag_chain.invoke(question)

    # Extract unique source files
    sources = []
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)

    print(f"\n✅ Answer: {answer}")
    print(f"\n📁 Sources: {sources}")

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }