# 🧠 CodeMind — AI-Powered Codebase Q&A Assistant

Ask any question about your codebase in plain English.
Built with RAG · LangChain · ChromaDB · Groq LLM · FastAPI · Streamlit

---

## 🚀 How It Works

```
Your Code Files
      ↓
Split into chunks (LangChain)
      ↓
Convert to Embeddings (Sentence Transformers)
      ↓
Store in Vector Database (ChromaDB)
      ↓
User asks a question
      ↓
ChromaDB finds most relevant code chunks
      ↓
Groq LLM (Llama 3) reads chunks and answers
      ↓
Answer + Source File References
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| RAG Framework | LangChain |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | Groq (Llama 3 70B) |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Language | Python |

---

## 📁 Project Structure

```
codemind/
├── file_loader.py     # Step 1: Loads code files from folder
├── vector_store.py    # Step 2: Creates embeddings + stores in ChromaDB
├── rag_chain.py       # Step 3: Groq LLM + RAG chain
├── main.py            # Step 4: FastAPI REST API
├── app_ui.py          # Step 5: Streamlit UI
├── requirements.txt   # All dependencies
└── README.md          # This file
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free Groq API Key
- Go to https://console.groq.com
- Sign up for free
- Create an API key
- Copy the key

### 3. Set API Key
```bash
# Windows
set GROQ_API_KEY=your-groq-api-key-here

# Mac/Linux
export GROQ_API_KEY=your-groq-api-key-here
```

### 4. Run Streamlit UI
```bash
streamlit run app_ui.py
```

### 5. OR Run FastAPI Backend
```bash
uvicorn main:app --reload
```
Then open: http://localhost:8000/docs

---

## 💬 Example Questions

- "How does the loan prediction API work?"
- "What API endpoints are available?"
- "How is the ML model loaded?"
- "What does the predict function do?"
- "What libraries are used in this project?"
- "How is input validation handled?"

---

## 👨‍💻 Built By

Pradeep Kunapareddi
- LinkedIn: https://www.linkedin.com/in/sri-omkar-pradeep-kunapareddi-28aa202b7
- GitHub: https://github.com/PradeepKunapareddi
