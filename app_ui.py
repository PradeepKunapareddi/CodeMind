# ============================================================
# STEP 5 — STREAMLIT UI
#
# This is the frontend of CodeMind.
# A simple, clean web interface where:
# - User selects a code folder
# - Clicks "Index Codebase" to process files
# - Types questions and gets answers
#
# Run with: streamlit run app_ui.py
# ============================================================

import streamlit as st
import os
from file_loader import load_code_files
from vector_store import create_vector_store, load_existing_vector_store
from rag_chain import create_rag_chain, ask_question

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="CodeMind — AI Codebase Assistant",
    page_icon="🧠",
    layout="wide"
)

# ── HEADER ────────────────────────────────────────────────────
st.title("🧠 CodeMind")
st.subheader("AI-Powered Codebase Q&A Assistant")
st.markdown("*Ask any question about your code in plain English*")
st.divider()

# ── SIDEBAR — CONFIGURATION ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Groq API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key from https://console.groq.com"
    )

    st.divider()

    # Folder path input
    st.header("📂 Load Codebase")
    folder_path = st.text_input(
        "Code Folder Path",
        value="./sample_code",
        help="Enter the path to your code folder"
    )

    # Index button
    if st.button("🚀 Index Codebase", use_container_width=True):

        if not groq_api_key:
            st.error("Please enter your Groq API key first!")

        elif not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")

        else:
            with st.spinner("Loading and indexing your code..."):
                try:
                    # Step 1: Load files
                    st.info("📁 Loading code files...")
                    docs = load_code_files(folder_path)

                    if not docs:
                        st.error("No code files found in this folder!")
                    else:
                        # Step 2: Create vector store
                        st.info("🧠 Creating embeddings...")
                        vs = create_vector_store(docs)

                        # Step 3: Create RAG chain
                        st.info("⛓️ Building RAG chain...")
                        chain = create_rag_chain(vs, groq_api_key)

                        # Store in session state so it persists
                        st.session_state["rag_chain"] = chain
                        st.session_state["files_loaded"] = len(docs)
                        st.session_state["folder"] = folder_path

                        st.success(f"✅ {len(docs)} files indexed successfully!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show status
    if "rag_chain" in st.session_state:
        st.success(f"✅ Ready! {st.session_state.get('files_loaded', 0)} files loaded")
        st.caption(f"📂 {st.session_state.get('folder', '')}")
    else:
        st.warning("⚠️ No codebase loaded yet")

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Enter your Groq API key")
    st.markdown("2. Enter your code folder path")
    st.markdown("3. Click Index Codebase")
    st.markdown("4. Ask questions below!")


# ── MAIN AREA — Q&A ───────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask About Your Code")

    # Example questions
    st.markdown("**Example questions:**")
    example_questions = [
        "How does the loan prediction API work?",
        "What API endpoints are available?",
        "How is the ML model loaded and used?",
        "What does the predict function do?",
        "What libraries are used in this project?",
        "How is input validation handled?"
    ]

    # Quick question buttons
    cols = st.columns(2)
    for i, eq in enumerate(example_questions):
        if cols[i % 2].button(eq, key=f"eq_{i}", use_container_width=True):
            st.session_state["current_question"] = eq

    st.divider()

    # Question input
    question = st.text_area(
        "Your Question",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g. How does the prediction endpoint work?",
        height=100
    )

    # Ask button
    if st.button("🔍 Ask CodeMind", use_container_width=True, type="primary"):

        if "rag_chain" not in st.session_state:
            st.error("Please index a codebase first using the sidebar!")

        elif not question.strip():
            st.warning("Please enter a question!")

        else:
            with st.spinner("🧠 CodeMind is thinking..."):
                try:
                    result = ask_question(
                        st.session_state["rag_chain"],
                        question
                    )

                    # Store in chat history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []

                    st.session_state["history"].insert(0, result)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.header("📊 Stats")
    if "rag_chain" in st.session_state:
        st.metric("Files Indexed", st.session_state.get("files_loaded", 0))
        st.metric("Questions Asked", len(st.session_state.get("history", [])))
    else:
        st.info("Load a codebase to see stats")

# ── DISPLAY ANSWERS ───────────────────────────────────────────
st.divider()
st.header("📋 Answers")

if "history" not in st.session_state or not st.session_state["history"]:
    st.info("No questions asked yet. Ask something above!")
else:
    for i, item in enumerate(st.session_state["history"]):
        with st.expander(f"Q: {item['question']}", expanded=(i == 0)):

            # Answer
            st.markdown("**Answer:**")
            st.markdown(item["answer"])

            # Sources
            if item["sources"]:
                st.divider()
                st.markdown("**📁 Source Files Used:**")
                for source in item["sources"]:
                    st.code(source, language=None)
